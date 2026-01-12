#include "NSKernel.h"

namespace Hyres {

void NsKernel::calculate_viscous_rhs() {
    int rank = mpi_.get_rank();

    for (Block* b : blocks_) {
        // 1. 检查是否为本地 Block
        if (b && b->owner_rank == rank) {
            
            // 2. 调用核心计算逻辑 (针对单个 Block)
            compute_block_viscous_rhs(b);
        }
    }
}

void NsKernel::compute_block_viscous_rhs(Block* b) {
    const auto& global = GlobalData::getInstance();
    
    // -------------------------------------------------------------------------
    // 1. 准备常量与参数
    // -------------------------------------------------------------------------
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng; // Ghost layer count
    
    // nmax (通常取 max(ni,nj,nk) + 保护层)
    int nmax = std::max({ni, nj, nk}) + 2 * ng;

    real_t reynolds = global.reynolds;
    real_t mach     = global.mach;     // moo
    real_t gamma    = global.gamma;    // gama
    real_t prl      = global.pr_laminar;
    real_t pr_turb  = global.pr_turbulent;
    int    nvis     = config_->physics.viscous_mode; 

    real_t re_inv   = 1.0 / reynolds;
    
    // cp = 1.0 / ((gama-1.0) * moo * moo)
    real_t cp       = 1.0 / ((gamma - 1.0) * mach * mach);
    real_t cp_prl   = cp / prl;
    real_t cp_prt   = cp / pr_turb;

    // -------------------------------------------------------------------------
    // 2. 分配临时存储空间 (建议优化为 Block 成员变量)
    // -------------------------------------------------------------------------
    HyresArray<real_t> duvwt;
    duvwt.resize(ni + 2*ng, nj + 2*ng, nk + 2*ng, 12); 

    HyresArray<real_t> duvwt_mid;
    duvwt_mid.resize(ni + 2*ng, nj + 2*ng, nk + 2*ng, 12);

    // 1D Buffers
    std::vector<real_t> vslt1_line(nmax), vslt2_line(nmax);
    std::vector<real_t> vslt1_half(nmax), vslt2_half(nmax);
    std::vector<real_t> vol_line(nmax), vol_half(nmax);
    
    std::vector<std::vector<real_t>> kxyz_line(9, std::vector<real_t>(nmax));
    std::vector<std::vector<real_t>> kxyz_half(9, std::vector<real_t>(nmax));

    std::vector<std::vector<real_t>> duvwt_line(12, std::vector<real_t>(nmax));
    std::vector<std::vector<real_t>> duvwt_half(12, std::vector<real_t>(nmax));
    std::vector<std::vector<real_t>> duvwtdxyz(12, std::vector<real_t>(nmax)); 

    std::vector<std::vector<real_t>> uvwt_lvir(4, std::vector<real_t>(nmax + 10)); 
    std::vector<std::vector<real_t>> uvwt_half(4, std::vector<real_t>(nmax));

    std::vector<std::vector<real_t>> fv(5, std::vector<real_t>(nmax)); 
    std::vector<std::vector<real_t>> dfv(5, std::vector<real_t>(nmax));

    // -------------------------------------------------------------------------
    // 3. 计算全局导数
    // -------------------------------------------------------------------------
    uvwt_der_4th_virtual(b, duvwt);
    uvwt_der_4th_half_virtual(b, duvwt_mid);

    // -------------------------------------------------------------------------
    // 4. I-Direction Sweep (du/dxi provided by duvwt_mid)
    // -------------------------------------------------------------------------
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            
            int K = k + ng;
            int J = j + ng;
            int offset = 2; // Map -2 to 0

            // --- 4.1 提取 1D 数据 (i-loop) ---
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                
                // Viscosity
                real_t v_lam = b->visl(I, J, K);
                real_t v_tur = (nvis >= 1) ? b->vist(I, J, K) : 0.0;
                vslt1_line[i] = v_lam + v_tur;
                vslt2_line[i] = v_lam * cp_prl + v_tur * cp_prt;

                // Metrics (kcx...ctz)
                kxyz_line[0][i] = b->kcx(I, J, K); kxyz_line[1][i] = b->etx(I, J, K); kxyz_line[2][i] = b->ctx(I, J, K);
                kxyz_line[3][i] = b->kcy(I, J, K); kxyz_line[4][i] = b->ety(I, J, K); kxyz_line[5][i] = b->cty(I, J, K);
                kxyz_line[6][i] = b->kcz(I, J, K); kxyz_line[7][i] = b->etz(I, J, K); kxyz_line[8][i] = b->ctz(I, J, K);

                vol_line[i] = b->vol(I, J, K);

                // duvwt_line (m=4..11, Skip 0-3 which are I-derivs)
                for (int m = 4; m < 12; ++m) { 
                    duvwt_line[m][i] = duvwt(I, J, K, m);
                }
            }

            // --- 4.2 扩展变量与边界检查 (-2:ni+3) ---
            for (int i_idx = -2; i_idx < ni + 3; ++i_idx) {
                int I_block = i_idx + ng - 1; 
                if (I_block < 0) I_block = 0; // Safety

                uvwt_lvir[0][i_idx + offset] = b->u(I_block, J, K);
                uvwt_lvir[1][i_idx + offset] = b->v(I_block, J, K);
                uvwt_lvir[2][i_idx + offset] = b->w(I_block, J, K);
                uvwt_lvir[3][i_idx + offset] = b->t(I_block, J, K);
            }

            if (b->r(ng-3, J, K) < 0.0) {
                    uvwt_lvir[3][-2 + offset] = -1.0;
            }
            if (b->r(ni + ng + 2, J, K) < 0.0) {
                 uvwt_lvir[3][ni + 3 + offset] = -1.0;
            }

            // --- 4.3 插值 ---
            compute_value_line_half_virtual(nmax, ni, uvwt_lvir, uvwt_half, offset);
            
            // Mask: (0, 1, 1) -> Skip I-derivs (0-3)
            compute_value_half_node_ijk(nmax, ni, 12, 0, 1, 1, duvwt_line, duvwt_half);
            
            compute_value_half_node(nmax, ni, 9, kxyz_line, kxyz_half);
            compute_value_half_node_scalar(nmax, ni, vol_line, vol_half);
            compute_value_half_node_scalar(nmax, ni, vslt1_line, vslt1_half);
            compute_value_half_node_scalar(nmax, ni, vslt2_line, vslt2_half);

            // Fill I-derivs from duvwt_mid (0-3)
            // Note: duvwt_mid(I,...) where I=i+ng corresponds to interface i
            for (int i = 0; i <= ni; ++i) {
                int I_mid = i + ng - 1; // duvwt_mid uses 0-based ng-1 mapping
                for (int m = 0; m < 4; ++m) {
                    duvwt_half[m][i] = duvwt_mid(I_mid, J, K, m);
                }
            }

            // --- 4.4 计算物理导数 ---
            compute_duvwt_dxyz(nmax, ni, duvwt_half, kxyz_half, vol_half, duvwtdxyz);
            
            // --- 4.5 计算粘性通量 ---
            // Metric Indices: 0(kcx), 3(kcy), 6(kcz)
            compute_flux_vis_line_new(nmax, ni, uvwt_half, duvwtdxyz, kxyz_half, 
                                      0, 3, 6, 
                                      vslt1_half, vslt2_half, fv);

            // --- 4.6 通量导数与更新 ---
            compute_flux_dxyz(nmax, ni, 5, fv, dfv);

            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                for (int m = 0; m < 5; ++m) {
                    b->dq(I, J, K, m) -= re_inv * dfv[m][i];
                }
            }
        } 
    }

    // -------------------------------------------------------------------------
    // 5. J-Direction Sweep (du/deta provided by duvwt_mid)
    // -------------------------------------------------------------------------
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            
            int K = k + ng;
            int I = i + ng;
            int offset = 2;

            // --- 5.1 提取 1D 数据 (j-loop) ---
            for (int j = 0; j < nj; ++j) {
                int J = j + ng;

                real_t v_lam = b->visl(I, J, K);
                real_t v_tur = (nvis >= 1) ? b->vist(I, J, K) : 0.0;
                vslt1_line[j] = v_lam + v_tur;
                vslt2_line[j] = v_lam * cp_prl + v_tur * cp_prt;

                kxyz_line[0][j] = b->kcx(I, J, K); kxyz_line[1][j] = b->etx(I, J, K); kxyz_line[2][j] = b->ctx(I, J, K);
                kxyz_line[3][j] = b->kcy(I, J, K); kxyz_line[4][j] = b->ety(I, J, K); kxyz_line[5][j] = b->cty(I, J, K);
                kxyz_line[6][j] = b->kcz(I, J, K); kxyz_line[7][j] = b->etz(I, J, K); kxyz_line[8][j] = b->ctz(I, J, K);

                vol_line[j] = b->vol(I, J, K);

                // duvwt_line (Skip 4-7 which are J-derivs)
                for (int m = 0; m < 4; ++m) duvwt_line[m][j] = duvwt(I, J, K, m);
                for (int m = 8; m < 12; ++m) duvwt_line[m][j] = duvwt(I, J, K, m);
            }

            // --- 5.2 扩展变量 ---
            for (int j_idx = -2; j_idx < nj + 3; ++j_idx) {
                int J_block = j_idx + ng - 1;
                if (J_block < 0) J_block = 0;

                uvwt_lvir[0][j_idx + offset] = b->u(I, J_block, K);
                uvwt_lvir[1][j_idx + offset] = b->v(I, J_block, K);
                uvwt_lvir[2][j_idx + offset] = b->w(I, J_block, K);
                uvwt_lvir[3][j_idx + offset] = b->t(I, J_block, K);
            }

            if (b->r(I, ng-3, K) < 0.0) {
                    uvwt_lvir[3][-2 + offset] = -1.0;
            }
            if (b->r(I, nj + ng + 2, K) < 0.0) {
                 uvwt_lvir[3][nj + 3 + offset] = -1.0;
            }

            // --- 5.3 插值 ---
            compute_value_line_half_virtual(nmax, nj, uvwt_lvir, uvwt_half, offset);

            // Mask: (1, 0, 1) -> Skip J-derivs (4-7)
            compute_value_half_node_ijk(nmax, nj, 12, 1, 0, 1, duvwt_line, duvwt_half);

            compute_value_half_node(nmax, nj, 9, kxyz_line, kxyz_half);
            compute_value_half_node_scalar(nmax, nj, vol_line, vol_half);
            compute_value_half_node_scalar(nmax, nj, vslt1_line, vslt1_half);
            compute_value_half_node_scalar(nmax, nj, vslt2_line, vslt2_half);

            // Fill J-derivs from duvwt_mid (4-7)
            for (int j = 0; j <= nj; ++j) {
                int J_mid = j + ng - 1; 
                for (int m = 0; m < 4; ++m) {
                    duvwt_half[m + 4][j] = duvwt_mid(I, J_mid, K, m + 4);
                }
            }

            // --- 5.4 通量计算 ---
            compute_duvwt_dxyz(nmax, nj, duvwt_half, kxyz_half, vol_half, duvwtdxyz);

            // Metric Indices: 1(etx), 4(ety), 7(etz)
            compute_flux_vis_line_new(nmax, nj, uvwt_half, duvwtdxyz, kxyz_half, 
                                      1, 4, 7, 
                                      vslt1_half, vslt2_half, fv);

            compute_flux_dxyz(nmax, nj, 5, fv, dfv);

            for (int j = 0; j < nj; ++j) {
                int J = j + ng;
                for (int m = 0; m < 5; ++m) {
                    b->dq(I, J, K, m) -= re_inv * dfv[m][j];
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 6. K-Direction Sweep (du/dzeta provided by duvwt_mid)
    // -------------------------------------------------------------------------
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            
            int J = j + ng;
            int I = i + ng;
            int offset = 2;

            // --- 6.1 提取 1D 数据 (k-loop) ---
            for (int k = 0; k < nk; ++k) {
                int K = k + ng;

                real_t v_lam = b->visl(I, J, K);
                real_t v_tur = (nvis >= 1) ? b->vist(I, J, K) : 0.0;
                vslt1_line[k] = v_lam + v_tur;
                vslt2_line[k] = v_lam * cp_prl + v_tur * cp_prt;

                kxyz_line[0][k] = b->kcx(I, J, K); kxyz_line[1][k] = b->etx(I, J, K); kxyz_line[2][k] = b->ctx(I, J, K);
                kxyz_line[3][k] = b->kcy(I, J, K); kxyz_line[4][k] = b->ety(I, J, K); kxyz_line[5][k] = b->cty(I, J, K);
                kxyz_line[6][k] = b->kcz(I, J, K); kxyz_line[7][k] = b->etz(I, J, K); kxyz_line[8][k] = b->ctz(I, J, K);

                vol_line[k] = b->vol(I, J, K);

                // duvwt_line (Skip 8-11 which are K-derivs)
                for (int m = 0; m < 8; ++m) duvwt_line[m][k] = duvwt(I, J, K, m);
            }

            // --- 6.2 扩展变量 ---
            for (int k_idx = -2; k_idx < nk + 3; ++k_idx) {
                int K_block = k_idx + ng - 1;
                if (K_block < 0) K_block = 0;

                uvwt_lvir[0][k_idx + offset] = b->u(I, J, K_block);
                uvwt_lvir[1][k_idx + offset] = b->v(I, J, K_block);
                uvwt_lvir[2][k_idx + offset] = b->w(I, J, K_block);
                uvwt_lvir[3][k_idx + offset] = b->t(I, J, K_block);
            }

            if (b->r(I, J, ng-3) < 0.0) {
                uvwt_lvir[3][-2 + offset] = -1.0;
            }
            if (b->r(I, J, nk + ng + 2) < 0.0) {
                 uvwt_lvir[3][nk + 3 + offset] = -1.0;
            }

            // --- 6.3 插值 ---
            compute_value_line_half_virtual(nmax, nk, uvwt_lvir, uvwt_half, offset);

            // Mask: (1, 1, 0) -> Skip K-derivs (8-11)
            compute_value_half_node_ijk(nmax, nk, 12, 1, 1, 0, duvwt_line, duvwt_half);

            compute_value_half_node(nmax, nk, 9, kxyz_line, kxyz_half);
            compute_value_half_node_scalar(nmax, nk, vol_line, vol_half);
            compute_value_half_node_scalar(nmax, nk, vslt1_line, vslt1_half);
            compute_value_half_node_scalar(nmax, nk, vslt2_line, vslt2_half);

            // Fill K-derivs from duvwt_mid (8-11)
            for (int k = 0; k <= nk; ++k) {
                int K_mid = k + ng - 1; 
                for (int m = 0; m < 4; ++m) {
                    duvwt_half[m + 8][k] = duvwt_mid(I, J, K_mid, m + 8);
                }
            }

            // --- 6.4 通量计算 ---
            compute_duvwt_dxyz(nmax, nk, duvwt_half, kxyz_half, vol_half, duvwtdxyz);

            // Metric Indices: 2(ctx), 5(cty), 8(ctz)
            compute_flux_vis_line_new(nmax, nk, uvwt_half, duvwtdxyz, kxyz_half, 
                                      2, 5, 8, 
                                      vslt1_half, vslt2_half, fv);

            compute_flux_dxyz(nmax, nk, 5, fv, dfv);

            for (int k = 0; k < nk; ++k) {
                int K = k + ng;
                for (int m = 0; m < 5; ++m) {
                    b->dq(I, J, K, m) -= re_inv * dfv[m][k];
                }
            }
        }
    }
}

void NsKernel::uvwt_der_4th_virtual(Block* b, HyresArray<real_t>& duvwt) {
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;

    // 分配一维线缓存 (足够容纳 -2 到 max(ni,nj,nk)+3)
    int nmax = std::max({ni, nj, nk}) + 2 * ng; 
    int buffer_size = nmax + 10; 
    int offset = 2; // 使得 vector[index + offset] 可以访问逻辑上的 -2

    // 线程安全/局部缓存
    std::vector<std::array<real_t, 4>> uvwt_line(buffer_size);
    std::vector<std::array<real_t, 4>> dtem_line(buffer_size);

    // =====================================================================
    // 1. I Direction (du/dxi -> duvwt 0-3)
    // =====================================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int K = k + ng;
            int J = j + ng;
            int st, ed;

            // --- 严格复刻 Fortran 边界判断 ---
            // Fortran: if( r(-2,j,k) < 0.0 )
            // C++: -2 对应 ng - 3
            if (b->r(ng - 3, J, K) < 0.0) {
                uvwt_line[-2 + offset][3] = -1.0; // Set Flag
                st = 0;
            } else {
                st = -2;
            }

            // Fortran: if( r(ni+3,j,k) < 0.0 )
            // C++: ni+3 对应 ni + ng + 2
            if (b->r(ni + ng + 2, J, K) < 0.0) {
                uvwt_line[ni + 3 + offset][3] = -1.0; // Set Flag
                ed = ni + 1;
            } else {
                ed = ni + 3;
            }

            // --- 数据提取 ---
            for (int i_idx = st; i_idx <= ed; ++i_idx) {
                int I = i_idx + ng - 1; // Fortran 1 -> C++ ng
                // 简单的钳位，防止 i_idx=-2 且 st=-2 时 I 越界 (如果 ng<3)
                // 如果 ng>=3，这里实际上不需要，但保留以防万一
                if (I < 0) I = 0; 

                uvwt_line[i_idx + offset][0] = b->u(I, J, K);
                uvwt_line[i_idx + offset][1] = b->v(I, J, K);
                uvwt_line[i_idx + offset][2] = b->w(I, J, K);
                uvwt_line[i_idx + offset][3] = b->t(I, J, K);
            }

            // --- 计算导数 ---
            compute_duvwt_node_line_virtual(nmax, ni, uvwt_line, dtem_line, offset);

            // --- 存回结果 (0-3) ---
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                // Fortran dtem(1:4, i) -> 
                for (int m = 0; m < 4; ++m) {
                    duvwt(I, J, K, m) = dtem_line[i][m];
                }
            }
        }
    }

    // =====================================================================
    // 2. J Direction (du/deta -> duvwt 4-7)
    // =====================================================================
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            int K = k + ng;
            int I = i + ng;
            int st, ed;

            // Fortran: if( r(i,-2,k) < 0.0 )
            if (b->r(I, ng - 3, K) < 0.0) {
                uvwt_line[-2 + offset][3] = -1.0;
                st = 0;
            } else {
                st = -2;
            }

            // Fortran: if( r(i,nj+3,k) < 0.0 )
            if (b->r(I, nj + ng + 2, K) < 0.0) {
                uvwt_line[nj + 3 + offset][3] = -1.0;
                ed = nj + 1;
            } else {
                ed = nj + 3;
            }

            for (int j_idx = st; j_idx <= ed; ++j_idx) {
                int J = j_idx + ng - 1;
                if (J < 0) J = 0;
                uvwt_line[j_idx + offset][0] = b->u(I, J, K);
                uvwt_line[j_idx + offset][1] = b->v(I, J, K);
                uvwt_line[j_idx + offset][2] = b->w(I, J, K);
                uvwt_line[j_idx + offset][3] = b->t(I, J, K);
            }

            compute_duvwt_node_line_virtual(nmax, nj, uvwt_line, dtem_line, offset);

            for (int j = 0; j < nj; ++j) {
                int J = j + ng;
                for (int m = 0; m < 4; ++m) {
                    duvwt(I, J, K, m + 4) = dtem_line[j][m];
                }
            }
        }
    }

    // =====================================================================
    // 3. K Direction (du/dzeta -> duvwt 8-11)
    // =====================================================================
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            int J = j + ng;
            int I = i + ng;
            int st, ed;

            // Fortran: if( r(i,j,-2) < 0.0 )
            if (b->r(I, J, ng - 3) < 0.0) {
                uvwt_line[-2 + offset][3] = -1.0;
                st = 0;
            } else {
                st = -2;
            }

            // Fortran: if( r(i,j,nk+3) < 0.0 )
            if (b->r(I, J, nk + ng + 2) < 0.0) {
                uvwt_line[nk + 3 + offset][3] = -1.0;
                ed = nk + 1;
            } else {
                ed = nk + 3;
            }

            for (int k_idx = st; k_idx <= ed; ++k_idx) {
                int K = k_idx + ng - 1;
                if (K < 0) K = 0;
                uvwt_line[k_idx + offset][0] = b->u(I, J, K);
                uvwt_line[k_idx + offset][1] = b->v(I, J, K);
                uvwt_line[k_idx + offset][2] = b->w(I, J, K);
                uvwt_line[k_idx + offset][3] = b->t(I, J, K);
            }

            compute_duvwt_node_line_virtual(nmax, nk, uvwt_line, dtem_line, offset);

            for (int k = 0; k < nk; ++k) {
                int K = k + ng;
                for (int m = 0; m < 4; ++m) {
                    duvwt(I, J, K, m + 8) = dtem_line[k][m];
                }
            }
        }
    }
}

void NsKernel::uvwt_der_4th_half_virtual(Block* b, HyresArray<real_t>& duvwt_mid) {
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;

    // Buffer allocation
    int nmax = std::max({ni, nj, nk}) + 2 * ng; 
    int buffer_size = nmax + 10; 
    int offset = 2; // Offset for ghost access (-2 mapping)

    // 临时一维数组
    std::vector<std::array<real_t, 4>> uvwt_line(buffer_size);
    std::vector<std::array<real_t, 4>> dtem_line(buffer_size); 

    // =====================================================================
    // 1. I Direction (du/dxi -> duvwt_mid 0-3)
    // =====================================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int K = k + ng;
            int J = j + ng;
            int st, ed;

            // --- 边界检查 ---
            if (b->r(ng - 3, J, K) < 0.0) { // r(-2)
                uvwt_line[-2 + offset][3] = -1.0; st = 0;
            } else { st = -2; }

            if (b->r(ni + ng + 2, J, K) < 0.0) { // r(ni+3)
                uvwt_line[ni + 3 + offset][3] = -1.0; ed = ni + 1;
            } else { ed = ni + 3; }

            // --- 数据提取 ---
            for (int i_idx = st; i_idx <= ed; ++i_idx) {
                int I = i_idx + ng - 1; 
                if (I < 0) I = 0; // Safety clamping
                uvwt_line[i_idx + offset][0] = b->u(I, J, K);
                uvwt_line[i_idx + offset][1] = b->v(I, J, K);
                uvwt_line[i_idx + offset][2] = b->w(I, J, K);
                uvwt_line[i_idx + offset][3] = b->t(I, J, K);
            }

            // --- 计算半点导数 ---
            compute_duvwt_half_line_virtual(nmax, ni, uvwt_line, dtem_line, offset);

            // --- 存回 duvwt_mid (Components 0-3) ---
            // Fortran: do i=0,ni ...
            // C++ Mapping: i -> i + ng - 1 (因为 duvwt_mid 的 ng-1 对应 Fortran 0)
            for (int i = 0; i <= ni; ++i) {
                int I_mid = i + ng - 1; 
                for (int m = 0; m < 4; ++m) {
                    duvwt_mid(I_mid, J, K, m) = dtem_line[i][m];
                }
            }
        }
    }

    // =====================================================================
    // 2. J Direction (du/deta -> duvwt_mid 4-7)
    // =====================================================================
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            int K = k + ng;
            int I = i + ng;
            int st, ed;

            // --- 边界检查 ---
            if (b->r(I, ng - 3, K) < 0.0) { 
                uvwt_line[-2 + offset][3] = -1.0; st = 0;
            } else { st = -2; }

            if (b->r(I, nj + ng + 2, K) < 0.0) {
                uvwt_line[nj + 3 + offset][3] = -1.0; ed = nj + 1;
            } else { ed = nj + 3; }

            // --- 数据提取 ---
            for (int j_idx = st; j_idx <= ed; ++j_idx) {
                int J = j_idx + ng - 1;
                if (J < 0) J = 0;
                uvwt_line[j_idx + offset][0] = b->u(I, J, K);
                uvwt_line[j_idx + offset][1] = b->v(I, J, K);
                uvwt_line[j_idx + offset][2] = b->w(I, J, K);
                uvwt_line[j_idx + offset][3] = b->t(I, J, K);
            }

            // --- 计算 ---
            compute_duvwt_half_line_virtual(nmax, nj, uvwt_line, dtem_line, offset);

            // --- 存回结果 (Components 4-7) ---
            // Fortran: do j=0,nj
            for (int j = 0; j <= nj; ++j) {
                int J_mid = j + ng - 1;
                for (int m = 0; m < 4; ++m) {
                    duvwt_mid(I, J_mid, K, m + 4) = dtem_line[j][m];
                }
            }
        }
    }

    // =====================================================================
    // 3. K Direction (du/dzeta -> duvwt_mid 8-11)
    // =====================================================================
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            int J = j + ng;
            int I = i + ng;
            int st, ed;

            // --- 边界检查 ---
            if (b->r(I, J, ng - 3) < 0.0) {
                uvwt_line[-2 + offset][3] = -1.0; st = 0;
            } else { st = -2; }

            if (b->r(I, J, nk + ng + 2) < 0.0) {
                uvwt_line[nk + 3 + offset][3] = -1.0; ed = nk + 1;
            } else { ed = nk + 3; }

            // --- 数据提取 ---
            for (int k_idx = st; k_idx <= ed; ++k_idx) {
                int K = k_idx + ng - 1;
                if (K < 0) K = 0;
                uvwt_line[k_idx + offset][0] = b->u(I, J, K);
                uvwt_line[k_idx + offset][1] = b->v(I, J, K);
                uvwt_line[k_idx + offset][2] = b->w(I, J, K);
                uvwt_line[k_idx + offset][3] = b->t(I, J, K);
            }

            // --- 计算 ---
            compute_duvwt_half_line_virtual(nmax, nk, uvwt_line, dtem_line, offset);

            // --- 存回结果 (Components 8-11) ---
            // Fortran: do k=0,nk
            for (int k = 0; k <= nk; ++k) {
                int K_mid = k + ng - 1;
                for (int m = 0; m < 4; ++m) {
                    duvwt_mid(I, J, K_mid, m + 8) = dtem_line[k][m];
                }
            }
        }
    }
}

void NsKernel::compute_duvwt_node_line_virtual(
    int nmax, int ni,
    const std::vector<std::array<real_t, 4>>& uvwt, 
    std::vector<std::array<real_t, 4>>& duvwt,
    int offset
) {
    // -------------------------------------------------------------------------
    // 1. 系数定义 (严格对应 Fortran DUVWT_NODE_LINE_virtual)
    // -------------------------------------------------------------------------
    // 内部点 (Central 4th)
    const real_t A1 =  8.0;
    const real_t B1 = -1.0;
    const real_t DD12_INV = 1.0 / 12.0;
    
    // 边界点 i=1 系数 (3rd Order Biased)
    const real_t DD6_INV = 1.0 / 6.0;
    // B11=-11, B12=18, B13=-9, B14=2 (硬编码在逻辑中)

    // 边界点 i=2 系数 (4th Order Biased)
    const real_t A2 = -3.0;
    const real_t B2 = -10.0;
    const real_t C2 = 18.0;
    const real_t D2 = -6.0;
    const real_t E2 = 1.0;

    int st = 1;  // Fortran loop start (1-based)
    int ed = ni; // Fortran loop end (1-based)

    // -------------------------------------------------------------------------
    // 2. 左边界处理 (Left Boundary)
    // 检查 uvwt(4, -2) < 0.0 -> C++: uvwt[-2 + offset][3]
    // -------------------------------------------------------------------------
    if (uvwt[-2 + offset][3] < 0.0) {
        st = 3; // 内部循环从 3 开始

        // i = 1 (C++ index 0)
        // Fortran: (-11*u1 + 18*u2 - 9*u3 + 2*u4)/6
        for (int m = 0; m < 4; ++m) {
            duvwt[0][m] = ( -11.0 * uvwt[1 + offset][m] 
                            + 18.0 * uvwt[2 + offset][m] 
                            -  9.0 * uvwt[3 + offset][m] 
                            +  2.0 * uvwt[4 + offset][m] ) * DD6_INV;
        }

        // i = 2 (C++ index 1)
        // Fortran: (A2*u1 + B2*u2 + C2*u3 + D2*u4 + E2*u5)/12
        for (int m = 0; m < 4; ++m) {
            duvwt[1][m] = ( A2 * uvwt[1 + offset][m] 
                          + B2 * uvwt[2 + offset][m] 
                          + C2 * uvwt[3 + offset][m] 
                          + D2 * uvwt[4 + offset][m] 
                          + E2 * uvwt[5 + offset][m] ) * DD12_INV;
        }
    }

    // -------------------------------------------------------------------------
    // 3. 右边界处理 (Right Boundary)
    // 检查 uvwt(4, ni+3) < 0.0 -> C++: uvwt[ni + 3 + offset][3]
    // -------------------------------------------------------------------------
    if (uvwt[ni + 3 + offset][3] < 0.0) {
        ed = ni - 2; // 内部循环到 ni-2 结束

        // i = ni (C++ index ni-1)
        // Fortran: -(-11*u_ni + 18*u_ni-1 - 9*u_ni-2 + 2*u_ni-3)/6
        for (int m = 0; m < 4; ++m) {
            duvwt[ni - 1][m] = -( -11.0 * uvwt[ni + offset][m] 
                                  + 18.0 * uvwt[ni - 1 + offset][m] 
                                  -  9.0 * uvwt[ni - 2 + offset][m] 
                                  +  2.0 * uvwt[ni - 3 + offset][m] ) * DD6_INV;
        }

        // i = ni-1 (C++ index ni-2)
        // Fortran: -(A2*u_ni + B2*u_ni-1 + ...)/12
        for (int m = 0; m < 4; ++m) {
            duvwt[ni - 2][m] = -( A2 * uvwt[ni + offset][m] 
                                + B2 * uvwt[ni - 1 + offset][m] 
                                + C2 * uvwt[ni - 2 + offset][m] 
                                + D2 * uvwt[ni - 3 + offset][m] 
                                + E2 * uvwt[ni - 4 + offset][m] ) * DD12_INV;
        }
    }

    // -------------------------------------------------------------------------
    // 4. 内部循环 (Central 4th Order)
    // Fortran: do i=st, ed ...
    // C++ indices: i-1 (from st-1 to ed-1)
    // Formula: (A1*(u(i+1)-u(i-1)) + B1*(u(i+2)-u(i-2)))/12
    // -------------------------------------------------------------------------
    for (int i = st; i <= ed; ++i) {
        // C++ output index: i-1
        int idx = i - 1; 
        for (int m = 0; m < 4; ++m) {
            duvwt[idx][m] = ( A1 * (uvwt[i + 1 + offset][m] - uvwt[i - 1 + offset][m]) 
                            + B1 * (uvwt[i + 2 + offset][m] - uvwt[i - 2 + offset][m]) ) * DD12_INV;
        }
    }
}

// =========================================================================
// 计算半点导数的一维核心算法 (对应 Fortran: subroutine DUVWT_half_line_virtual)
// =========================================================================
// 参数说明:
// nmax:   最大维度限制
// ni:     当前线上的物理点数
// uvwt:   输入变量 (u,v,w,t)，包含 Ghost 区域
//         索引映射: uvwt[k + offset] 对应 Fortran 的 uvwt(:, k)
// duvwt:  输出导数 (定义在半点 i+1/2 上)
//         索引映射: duvwt[i] 对应 Fortran 的 duvwt(:, i) (i=0..ni)
// offset: 偏移量，默认为 2
// =========================================================================
void NsKernel::compute_duvwt_half_line_virtual(
    int nmax, int ni,
    const std::vector<std::array<real_t, 4>>& uvwt, 
    std::vector<std::array<real_t, 4>>& duvwt,
    int offset
) {
    // -------------------------------------------------------------------------
    // 1. 定义差分系数
    // -------------------------------------------------------------------------
    // 中央部分 (Central) - 4阶显式近似 (Explicit 4th order for half point)
    const real_t A1 = 27.0, B1 = -1.0;
    const real_t dd24_inv = 1.0 / 24.0;

    // 左边界偏心格式 (Left Biased)
    // i=1 (4th/5th order)
    const real_t A2 = -22.0, B2 = 17.0, C2 = 9.0, D2 = -5.0, E2 = 1.0;
    // i=0 (3rd/4th order)
    const real_t A3 = -71.0, B3 = 141.0, C3 = -93.0, D3 = 23.0;

    int st = 0;  // Fortran loop start
    int ed = ni; // Fortran loop end

    // -------------------------------------------------------------------------
    // 2. 左边界检查 (Left Boundary Check)
    // -------------------------------------------------------------------------
    // 检查 uvwt[-2] 的 Flag (Temperature < 0)
    if (uvwt[-2 + offset][3] < 0.0) {
        st = 2; // 主循环从 2 开始

        // i = 0 (半点): 偏心差分
        // Fortran: (A3*uvwt(1) + B3*uvwt(2) + C3*uvwt(3) + D3*uvwt(4)) / 24
        for (int m = 0; m < 4; ++m) {
            duvwt[0][m] = ( A3 * uvwt[1 + offset][m] 
                          + B3 * uvwt[2 + offset][m] 
                          + C3 * uvwt[3 + offset][m] 
                          + D3 * uvwt[4 + offset][m] ) * dd24_inv;
        }

        // i = 1 (半点): 偏心差分
        // Fortran: (A2*uvwt(1) + B2*uvwt(2) + C2*uvwt(3) + D2*uvwt(4) + E2*uvwt(5)) / 24
        for (int m = 0; m < 4; ++m) {
            duvwt[1][m] = ( A2 * uvwt[1 + offset][m] 
                          + B2 * uvwt[2 + offset][m] 
                          + C2 * uvwt[3 + offset][m] 
                          + D2 * uvwt[4 + offset][m] 
                          + E2 * uvwt[5 + offset][m] ) * dd24_inv;
        }
    }

    // -------------------------------------------------------------------------
    // 3. 右边界检查 (Right Boundary Check)
    // -------------------------------------------------------------------------
    // 检查 uvwt[ni+3] 的 Flag
    if (uvwt[ni + 3 + offset][3] < 0.0) {
        ed = ni - 2; // 主循环结束于 ni-2

        // i = ni (半点): 偏心差分 (Backward)
        // Fortran: -(A3*uvwt(ni) + B3*uvwt(ni-1) + C3*uvwt(ni-2) + D3*uvwt(ni-3)) / 24
        for (int m = 0; m < 4; ++m) {
            duvwt[ni][m] = -( A3 * uvwt[ni + offset][m] 
                            + B3 * uvwt[ni - 1 + offset][m] 
                            + C3 * uvwt[ni - 2 + offset][m] 
                            + D3 * uvwt[ni - 3 + offset][m] ) * dd24_inv;
        }

        // i = ni - 1 (半点): 偏心差分 (Backward)
        // Fortran: -(A2*uvwt(ni) + B2*uvwt(ni-1) + ... + E2*uvwt(ni-4)) / 24
        for (int m = 0; m < 4; ++m) {
            duvwt[ni - 1][m] = -( A2 * uvwt[ni + offset][m] 
                                + B2 * uvwt[ni - 1 + offset][m] 
                                + C2 * uvwt[ni - 2 + offset][m] 
                                + D2 * uvwt[ni - 3 + offset][m] 
                                + E2 * uvwt[ni - 4 + offset][m] ) * dd24_inv;
        }
    }

    // -------------------------------------------------------------------------
    // 4. 内部主循环 (Compact-like Central Difference for Half Point)
    // -------------------------------------------------------------------------
    // Fortran: duvwt(m,i) = ( A1*(uvwt(m,i+1) - uvwt(m,i)) + B1*(uvwt(m,i+2) - uvwt(m,i-1)) ) / 24
    // 这里的 i 代表半点 i+1/2，涉及到的节点是 i, i+1 (内层) 和 i-1, i+2 (外层)
    for (int i = st; i <= ed; ++i) {
        for (int m = 0; m < 4; ++m) {
            duvwt[i][m] = ( A1 * (uvwt[i + 1 + offset][m] - uvwt[i + offset][m]) 
                          + B1 * (uvwt[i + 2 + offset][m] - uvwt[i - 1 + offset][m]) ) * dd24_inv;
        }
    }
}

// =========================================================================
// 原始变量插值到半点 (修正版：适配 vector<vector> SoA 布局)
// Input q: [component][index]
// =========================================================================
void NsKernel::compute_value_line_half_virtual(
    int nmax, int ni,
    const std::vector<std::vector<real_t>>& q, 
    std::vector<std::vector<real_t>>& q_half,
    int offset
) {
    const real_t A1 = 9.0, B1 = -1.0;
    const real_t dd16_inv = 1.0 / 16.0;
    const real_t A2 = 5.0, B2 = 15.0, C2 = -5.0, D2 = 1.0;
    const real_t A3 = 35.0, B3 = -35.0, C3 = 21.0, D3 = -5.0;

    int st = 0;
    int ed = ni;

    // 左边界检查: 检查 Component 3 (T) 在索引 -2 处的值
    if (q[3][-2 + offset] < 0.0) {
        st = 2;
        for (int m = 0; m < 4; ++m) {
            q_half[m][0] = (A3 * q[m][1 + offset] + B3 * q[m][2 + offset] + C3 * q[m][3 + offset] + D3 * q[m][4 + offset]) * dd16_inv;
            q_half[m][1] = (A2 * q[m][1 + offset] + B2 * q[m][2 + offset] + C2 * q[m][3 + offset] + D2 * q[m][4 + offset]) * dd16_inv;
        }
    }

    // 右边界检查
    if (q[3][ni + 3 + offset] < 0.0) {
        ed = ni - 2;
        for (int m = 0; m < 4; ++m) {
            q_half[m][ni]     = (A3 * q[m][ni + offset] + B3 * q[m][ni - 1 + offset] + C3 * q[m][ni - 2 + offset] + D3 * q[m][ni - 3 + offset]) * dd16_inv;
            q_half[m][ni - 1] = (A2 * q[m][ni + offset] + B2 * q[m][ni - 1 + offset] + C2 * q[m][ni - 2 + offset] + D2 * q[m][ni - 3 + offset]) * dd16_inv;
        }
    }

    // 内部循环
    for (int m = 0; m < 4; ++m) {
        for (int i = st; i <= ed; ++i) {
            q_half[m][i] = (A1 * (q[m][i + offset] + q[m][i + 1 + offset]) + 
                            B1 * (q[m][i + 2 + offset] + q[m][i - 1 + offset])) * dd16_inv;
        }
    }
}

// =========================================================================
// 导数变量半点插值 (对应 Fortran: subroutine VALUE_HALF_NODE_IJK)
// 支持分量掩码 (IP, JP, KP)
// =========================================================================
void NsKernel::compute_value_half_node_ijk(
    int nmax, int ni, int n_comps,
    int ip, int jp, int kp, // 掩码标志
    const std::vector<std::vector<real_t>>& q, // Input: [comp][index]
    std::vector<std::vector<real_t>>& q_half   // Output: [comp][index]
) {
    // -------------------------------------------------------------------------
    // 1. 确定需要处理的分量范围
    // -------------------------------------------------------------------------
    int n3 = n_comps / 3; // 对应 Fortran 的 N3 = N/3 (通常是 4)
    
    // 我们将需要处理的分量索引放入列表
    std::vector<int> active_comps;
    active_comps.reserve(n_comps);

    // IP 对应前 N3 个分量 (0 到 n3-1)
    if (ip > 0) {
        for (int m = 0; m < n3; ++m) active_comps.push_back(m);
    }
    // JP 对应中间 N3 个分量 (n3 到 2*n3-1)
    if (jp > 0) {
        for (int m = n3; m < 2 * n3; ++m) active_comps.push_back(m);
    }
    // KP 对应后 N3 个分量 (2*n3 到 n_comps-1)
    if (kp > 0) {
        for (int m = 2 * n3; m < n_comps; ++m) active_comps.push_back(m);
    }

    // -------------------------------------------------------------------------
    // 2. 定义插值系数
    // -------------------------------------------------------------------------
    const real_t A1 = 9.0, B1 = -1.0;
    const real_t dd16_inv = 1.0 / 16.0;

    // 边界系数 (i=1, ni-1)
    const real_t A2 = 5.0, B2 = 15.0, C2 = -5.0, D2 = 1.0;
    // 边界系数 (i=0, ni)
    const real_t A3 = 35.0, B3 = -35.0, C3 = 21.0, D3 = -5.0;

    // -------------------------------------------------------------------------
    // 3. 执行插值
    // Input q 是 0-based (0..ni-1), 对应 Fortran 1..ni
    // Output q_half 是 0-based (0..ni), 对应 Fortran 0..ni
    // -------------------------------------------------------------------------
    
    // 遍历所有激活的分量
    for (int m : active_comps) {
        
        // --- 内部循环 (Central 4th Order) ---
        // Range: i from 2 to ni-2 (Fortran) -> 1 to ni-3 (C++ index of q)
        // 插值点: q_half[i] 对应 Fortran i
        // 依赖: q[i], q[i+1], q[i+2], q[i-1] (Fortran indices)
        // C++ 映射: q_fortran(k) -> q_cpp[k-1]
        for (int i = 2; i <= ni - 2; ++i) {
            real_t q_i   = q[m][i - 1]; // q(i)
            real_t q_ip1 = q[m][i];     // q(i+1)
            real_t q_ip2 = q[m][i + 1]; // q(i+2)
            real_t q_im1 = q[m][i - 2]; // q(i-1)

            q_half[m][i] = (A1 * (q_i + q_ip1) + B1 * (q_ip2 + q_im1)) * dd16_inv;
        }

        // --- 边界处理 ---
        
        // i = 1
        q_half[m][1] = (A2 * q[m][0] + B2 * q[m][1] + C2 * q[m][2] + D2 * q[m][3]) * dd16_inv;

        // i = ni - 1
        // indices: ni, ni-1, ni-2, ni-3 (Fortran) -> ni-1, ni-2, ni-3, ni-4 (C++)
        q_half[m][ni - 1] = (A2 * q[m][ni - 1] + B2 * q[m][ni - 2] + C2 * q[m][ni - 3] + D2 * q[m][ni - 4]) * dd16_inv;

        // i = 0
        q_half[m][0] = (A3 * q[m][0] + B3 * q[m][1] + C3 * q[m][2] + D3 * q[m][3]) * dd16_inv;

        // i = ni
        q_half[m][ni] = (A3 * q[m][ni - 1] + B3 * q[m][ni - 2] + C3 * q[m][ni - 3] + D3 * q[m][ni - 4]) * dd16_inv;
    }
}

// =========================================================================
// 通用半点插值 (对应 Fortran: subroutine VALUE_HALF_NODE)
// 用于 Metric, Vol, Viscosity 等
// =========================================================================
void NsKernel::compute_value_half_node(
    int nmax, int ni, int n_comps,
    const std::vector<std::vector<real_t>>& q, 
    std::vector<std::vector<real_t>>& q_half
) {
    // 复用 IJK 版本的逻辑，将 IP, JP, KP 全部设为 1 (或其他非0值)
    // 假设 n_comps 是 3 的倍数 (如 9, 12)，这在当前上下文中成立
    // 如果 n_comps 不是 3 的倍数 (如 1)，我们需要特殊处理
    
    if (n_comps % 3 == 0) {
        compute_value_half_node_ijk(nmax, ni, n_comps, 1, 1, 1, q, q_half);
    } else {
        // 对于 n_comps = 1 (如 vol, vis) 或其他情况，手动全量处理
        // 实际上可以直接把上面 compute_value_half_node_ijk 的内部逻辑复制一遍，
        // 去掉 active_comps 的筛选，直接遍历 m = 0 到 n_comps-1
        
        const real_t A1 = 9.0, B1 = -1.0;
        const real_t dd16_inv = 1.0 / 16.0;
        const real_t A2 = 5.0, B2 = 15.0, C2 = -5.0, D2 = 1.0;
        const real_t A3 = 35.0, B3 = -35.0, C3 = 21.0, D3 = -5.0;

        for (int m = 0; m < n_comps; ++m) {
            for (int i = 2; i <= ni - 2; ++i) {
                q_half[m][i] = (A1 * (q[m][i - 1] + q[m][i]) + B1 * (q[m][i + 1] + q[m][i - 2])) * dd16_inv;
            }
            q_half[m][1]      = (A2 * q[m][0] + B2 * q[m][1] + C2 * q[m][2] + D2 * q[m][3]) * dd16_inv;
            q_half[m][ni - 1] = (A2 * q[m][ni - 1] + B2 * q[m][ni - 2] + C2 * q[m][ni - 3] + D2 * q[m][ni - 4]) * dd16_inv;
            q_half[m][0]      = (A3 * q[m][0] + B3 * q[m][1] + C3 * q[m][2] + D3 * q[m][3]) * dd16_inv;
            q_half[m][ni]     = (A3 * q[m][ni - 1] + B3 * q[m][ni - 2] + C3 * q[m][ni - 3] + D3 * q[m][ni - 4]) * dd16_inv;
        }
    }
}

// =========================================================================
// 标量半点插值 (新增：用于处理一维 vector)
// =========================================================================
void NsKernel::compute_value_half_node_scalar(
    int nmax, int ni,
    const std::vector<real_t>& q, 
    std::vector<real_t>& q_half
) {
    const real_t A1 = 9.0, B1 = -1.0;
    const real_t dd16_inv = 1.0 / 16.0;
    const real_t A2 = 5.0, B2 = 15.0, C2 = -5.0, D2 = 1.0;
    const real_t A3 = 35.0, B3 = -35.0, C3 = 21.0, D3 = -5.0;

    // 内部点
    for (int i = 2; i <= ni - 2; ++i) {
        q_half[i] = (A1 * (q[i - 1] + q[i]) + B1 * (q[i + 1] + q[i - 2])) * dd16_inv;
    }
    // 边界点
    q_half[1]      = (A2 * q[0] + B2 * q[1] + C2 * q[2] + D2 * q[3]) * dd16_inv;
    q_half[ni - 1] = (A2 * q[ni - 1] + B2 * q[ni - 2] + C2 * q[ni - 3] + D2 * q[ni - 4]) * dd16_inv;
    q_half[0]      = (A3 * q[0] + B3 * q[1] + C3 * q[2] + D3 * q[3]) * dd16_inv;
    q_half[ni]     = (A3 * q[ni - 1] + B3 * q[ni - 2] + C3 * q[ni - 3] + D3 * q[ni - 4]) * dd16_inv;
}

// =========================================================================
// 计算物理导数 (对应 Fortran: subroutine DUVWT_DXYZ)
// 利用链式法则: d/dx = (d/dxi * kcx + d/deta * etx + d/dzeta * ctx) / vol
// =========================================================================
// 参数:
// nmax, ni: 维度
// duvwt:    输入，计算空间导数 (半点值) [12][i]
//           Mapping: 0-3 (d/dxi), 4-7 (d/deta), 8-11 (d/dzeta)
// kxyz:     输入，度量系数 (半点值) [9][i]
//           Mapping: 0 (kcx), 1 (etx), 2 (ctx), 3 (kcy)...
// vol:      输入，体积 (半点值) [i]
// duvwtdxyz: 输出，物理空间导数 [12][i]
//           Mapping: 0-3 (d/dx), 4-7 (d/dy), 8-11 (d/dz)
// =========================================================================
void NsKernel::compute_duvwt_dxyz(
    int nmax, int ni,
    const std::vector<std::vector<real_t>>& duvwt,
    const std::vector<std::vector<real_t>>& kxyz,
    const std::vector<real_t>& vol,
    std::vector<std::vector<real_t>>& duvwtdxyz
) {
    // 遍历所有半点 (0 到 ni)
    for (int i = 0; i <= ni; ++i) {
        real_t inv_vol = 1.0 / vol[i];

        // ---------------------------------------------------------------------
        // m 循环对应物理坐标方向: 0->x, 1->y, 2->z
        // 对应 Fortran 的 m=1,2,3
        // ---------------------------------------------------------------------
        for (int m = 0; m < 3; ++m) {
            
            // 度量系数索引 (Metric Indices)
            // Fortran: m1 = m+m+m-2 (1->1, 2->4, 3->7)
            // C++ (0-based): 
            // m=0 (x) -> indices 0, 1, 2 (kcx, etx, ctx)
            // m=1 (y) -> indices 3, 4, 5 (kcy, ety, cty)
            // m=2 (z) -> indices 6, 7, 8 (kcz, etz, ctz)
            int idx_k = 3 * m;     // kcx / kcy / kcz
            int idx_e = 3 * m + 1; // etx / ety / etz
            int idx_c = 3 * m + 2; // ctx / cty / ctz

            // 计算该方向上的物理导数 (d/dx, d/dy, d/dz)
            // 针对 4 个变量: u, v, w, T
            
            // 变量 1: u (duvwt indices 0, 4, 8) -> 输出 duvwtdxyz 0, 4, 8 ?
            // 等等，Fortran 代码逻辑是：
            // duvwtdxyz(m, i)   = ... duvwt(1)..duvwt(5)..duvwt(9)  -> du/dx, du/dy, du/dz
            // duvwtdxyz(m+3, i) = ... duvwt(2)..duvwt(6)..duvwt(10) -> dv/dx, dv/dy, dv/dz
            // duvwtdxyz(m+6, i) = ... duvwt(3)..duvwt(7)..duvwt(11) -> dw/dx, dw/dy, dw/dz
            // duvwtdxyz(m+9, i) = ... duvwt(4)..duvwt(8)..duvwt(12) -> dT/dx, dT/dy, dT/dz
            
            // C++ 对应:
            // Output Index Offset: m (0,1,2)
            
            // 1. u 的导数 (Input indices: 0, 4, 8) -> Output: 0, 1, 2
            duvwtdxyz[m][i] = (kxyz[idx_k][i] * duvwt[0][i] + 
                               kxyz[idx_e][i] * duvwt[4][i] + 
                               kxyz[idx_c][i] * duvwt[8][i]) * inv_vol;

            // 2. v 的导数 (Input indices: 1, 5, 9) -> Output: 3, 4, 5
            duvwtdxyz[m + 3][i] = (kxyz[idx_k][i] * duvwt[1][i] + 
                                   kxyz[idx_e][i] * duvwt[5][i] + 
                                   kxyz[idx_c][i] * duvwt[9][i]) * inv_vol;

            // 3. w 的导数 (Input indices: 2, 6, 10) -> Output: 6, 7, 8
            duvwtdxyz[m + 6][i] = (kxyz[idx_k][i] * duvwt[2][i] + 
                                   kxyz[idx_e][i] * duvwt[6][i] + 
                                   kxyz[idx_c][i] * duvwt[10][i]) * inv_vol;

            // 4. T 的导数 (Input indices: 3, 7, 11) -> Output: 9, 10, 11
            duvwtdxyz[m + 9][i] = (kxyz[idx_k][i] * duvwt[3][i] + 
                                   kxyz[idx_e][i] * duvwt[7][i] + 
                                   kxyz[idx_c][i] * duvwt[11][i]) * inv_vol;
        }
    }
}

// =========================================================================
// 计算粘性通量 (修正版：适配 vector<vector> SoA 布局)
// Input uvwt: [component][index]
// =========================================================================
void NsKernel::compute_flux_vis_line_new(
    int nmax, int ni,
    const std::vector<std::vector<real_t>>& uvwt,
    const std::vector<std::vector<real_t>>& duvwtdxyz,
    const std::vector<std::vector<real_t>>& kxyz,
    int idx_nx, int idx_ny, int idx_nz,
    const std::vector<real_t>& vslt1,
    const std::vector<real_t>& vslt2,
    std::vector<std::vector<real_t>>& fv
) {
    const real_t CC = 2.0 / 3.0;

    for (int i = 0; i <= ni; ++i) {
        real_t vs = vslt1[i];
        real_t vscc = vs * CC;
        real_t kcp = vslt2[i];

        real_t dudx = duvwtdxyz[0][i]; real_t dudy = duvwtdxyz[1][i]; real_t dudz = duvwtdxyz[2][i];
        real_t dvdx = duvwtdxyz[3][i]; real_t dvdy = duvwtdxyz[4][i]; real_t dvdz = duvwtdxyz[5][i];
        real_t dwdx = duvwtdxyz[6][i]; real_t dwdy = duvwtdxyz[7][i]; real_t dwdz = duvwtdxyz[8][i];

        real_t txyz_1 = vscc * (2.0 * dudx - dvdy - dwdz);
        real_t txyz_5 = vscc * (2.0 * dvdy - dwdz - dudx);
        real_t txyz_9 = vscc * (2.0 * dwdz - dudx - dvdy);
        real_t txyz_2 = vs * (dudy + dvdx);
        real_t txyz_3 = vs * (dudz + dwdx);
        real_t txyz_6 = vs * (dvdz + dwdy);
        real_t txyz_4 = txyz_2;
        real_t txyz_7 = txyz_3;
        real_t txyz_8 = txyz_6;

        real_t nx = kxyz[idx_nx][i];
        real_t ny = kxyz[idx_ny][i];
        real_t nz = kxyz[idx_nz][i];

        fv[0][i] = 0.0;
        fv[1][i] = txyz_1 * nx + txyz_2 * ny + txyz_3 * nz;
        fv[2][i] = txyz_4 * nx + txyz_5 * ny + txyz_6 * nz;
        fv[3][i] = txyz_7 * nx + txyz_8 * ny + txyz_9 * nz;

        // 注意这里访问 uvwt 的方式变了: uvwt[comp][index]
        real_t work_term = uvwt[0][i] * fv[1][i] + 
                           uvwt[1][i] * fv[2][i] + 
                           uvwt[2][i] * fv[3][i];
        
        real_t heat_term = kcp * (duvwtdxyz[9][i] * nx + duvwtdxyz[10][i] * ny + duvwtdxyz[11][i] * nz);

        fv[4][i] = work_term + heat_term;
    }
}

// =========================================================================
// 计算通量导数
// =========================================================================
void NsKernel::compute_flux_dxyz(
    int nmax, int ni, int nl,
    const std::vector<std::vector<real_t>>& f,
    std::vector<std::vector<real_t>>& df
) {
    // -------------------------------------------------------------------------
    // 1. 系数定义
    // -------------------------------------------------------------------------
    const real_t AC = 2250.0;
    const real_t BC = -125.0;
    const real_t CC =    9.0;
    const real_t DC_INV = 1.0 / 1920.0;
    const real_t DD_INV = 1.0 / 24.0;

    // -------------------------------------------------------------------------
    // 2. 内部点 (Sixth-Order Central)
    // Fortran: i=3 to ni-2 => df(i)
    // C++: df[i-1] 对应 df(i). 
    // 让 k = i-1. 范围 k=2 to ni-3.
    // 公式映射: df(i) 用 f(i), f(i-1)... => df[k] 用 f[k+1], f[k]...
    // -------------------------------------------------------------------------
    for (int m = 0; m < nl; ++m) {
        for (int i = 2; i <= ni - 3; ++i) {
            // 【关键修正】所有的 f 索引都在原基础上 +1
            // 原 f[i]   -> f[i+1]
            // 原 f[i-1] -> f[i]
            // 原 f[i+1] -> f[i+2]
            // 原 f[i-2] -> f[i-1]
            // 原 f[i+2] -> f[i+3]
            // 原 f[i-3] -> f[i-2]
            
            df[m][i] = ( AC * (f[m][i + 1] - f[m][i]) 
                       + BC * (f[m][i + 2] - f[m][i - 1]) 
                       + CC * (f[m][i + 3] - f[m][i - 2]) ) * DC_INV;
        }
    }

    // -------------------------------------------------------------------------
    // 3. 边界点处理 (这部分之前的代码是正确的，因为是直接硬编码的索引)
    // -------------------------------------------------------------------------
    for (int m = 0; m < nl; ++m) {
        // i = 2 (Fortran) -> C++ 1
        // df(2) uses f(0)..f(3) -> f[0]..f[3]
        df[m][1] = (f[m][0] - 27.0 * f[m][1] + 27.0 * f[m][2] - f[m][3]) * DD_INV;

        // i = ni-1 (Fortran) -> C++ ni-2
        df[m][ni - 2] = -(f[m][ni] - 27.0 * f[m][ni - 1] + 27.0 * f[m][ni - 2] - f[m][ni - 3]) * DD_INV;

        // i = 1 (Fortran) -> C++ 0
        df[m][0] = (-23.0 * f[m][0] + 21.0 * f[m][1] + 3.0 * f[m][2] - f[m][3]) * DD_INV;

        // i = ni (Fortran) -> C++ ni-1
        df[m][ni - 1] = -(-23.0 * f[m][ni] + 21.0 * f[m][ni - 1] + 3.0 * f[m][ni - 2] - f[m][ni - 3]) * DD_INV;
    }
}

}