#include "NSKernel.h"

namespace Hyres {

// =========================================================================
// 辅助工具：Minmod 限制器
// =========================================================================
inline real_t minmod(real_t a, real_t b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

// =========================================================================
// 辅助工具：计算标准 Euler 通量 F(Q)
// F = [ rho*Vn, rho*u*Vn + nx*p, rho*v*Vn + ny*p, rho*w*Vn + nz*p, (rho*H)*Vn ]
// =========================================================================
// 参数:
//   q: 原始变量 [rho, u, v, w, p] (注意: 这里传入原始变量更方便计算)
//   nx, ny, nz: 界面法向量
//   gamma: 比热比
// 输出:
//   flux: 守恒通量 [5]
// =========================================================================
void compute_euler_flux(const real_t* q_prim, real_t nx, real_t ny, real_t nz, real_t gamma, real_t* flux) {
    real_t rho = q_prim[0];
    real_t u   = q_prim[1];
    real_t v   = q_prim[2];
    real_t w   = q_prim[3];
    real_t p   = q_prim[4];

    // 计算法向速度 Vn
    real_t vn = u * nx + v * ny + w * nz;
    
    // 计算总焓 H = (E + p) / rho = gamma/(gamma-1)*p/rho + 0.5*V^2
    real_t v2 = u*u + v*v + w*w;
    real_t H  = (gamma * p) / ((gamma - 1.0) * rho) + 0.5 * v2;

    flux[0] = rho * vn;
    flux[1] = rho * u * vn + nx * p;
    flux[2] = rho * v * vn + ny * p;
    flux[3] = rho * w * vn + nz * p;
    flux[4] = rho * H * vn;
}

// =========================================================================
// 静态辅助函数：直接复用您验证过的逻辑
// =========================================================================
static void value_half_node_static(int n, int len, const std::vector<real_t>& f, std::vector<real_t>& fh) {
    // 复刻您提供的 SimulationLoader::value_half_node
    int stride_f = len;
    int stride_fh = len + 1;

    for (int m = 0; m < n; ++m) {
        const real_t* q = &f[m * stride_f];
        real_t* q_half = &fh[m * stride_fh];

        for (int i = 2; i <= len - 2; ++i) {
            q_half[i] = (9.0 * (q[i-1] + q[i]) - (q[i-2] + q[i+1])) / 16.0;
        }
        q_half[1] = (5.0*q[0] + 15.0*q[1] - 5.0*q[2] + q[3]) / 16.0;
        q_half[len-1] = (5.0*q[len-1] + 15.0*q[len-2] - 5.0*q[len-3] + q[len-4]) / 16.0;
        q_half[0] = (35.0*q[0] - 35.0*q[1] + 21.0*q[2] - 5.0*q[3]) / 16.0;
        q_half[len] = (35.0*q[len-1] - 35.0*q[len-2] + 21.0*q[len-3] - 5.0*q[len-4]) / 16.0;
    }
}

static void flux_dxyz_static(int n, int len, const std::vector<real_t>& fh, std::vector<real_t>& df) {
    // 复刻 flux_dxyz
    int stride_fh = len + 1;
    int stride_df = len;

    for (int m = 0; m < n; ++m) {
        const real_t* f = &fh[m * stride_fh];
        real_t* d = &df[m * stride_df];

        for (int i = 2; i <= len - 3; ++i) {
            d[i] = (2250.0 * (f[i+1] - f[i]) - 125.0 * (f[i+2] - f[i-1]) + 9.0 * (f[i+3] - f[i-2])) / 1920.0;
        }
        d[1] = (f[0] - 27.0*f[1] + 27.0*f[2] - f[3]) / 24.0;
        d[len-2] = -(f[len] - 27.0*f[len-1] + 27.0*f[len-2] - f[len-3]) / 24.0;
        d[0] = (-23.0*f[0] + 21.0*f[1] + 3.0*f[2] - f[3]) / 24.0;
        d[len-1] = -(-23.0*f[len] + 21.0*f[len-1] + 3.0*f[len-2] - f[len-3]) / 24.0;
    }
}

void NsKernel::calculate_inviscous_rhs() {
    int rank = mpi_.get_rank();

    for (Block* b : blocks_) {
        // 1. 检查是否为本地 Block
        if (b && b->owner_rank == rank) {
            
            // 2. 调用核心计算逻辑 (针对单个 Block)
            compute_block_inviscous_rhs(b);
        }
    }
}

// =========================================================================
// 计算无粘右端项 (Inviscid RHS) - 框架实现
// 对应 Fortran: subroutine inviscd3d
// =========================================================================
void NsKernel::compute_block_inviscous_rhs(Block* b) {
    // -------------------------------------------------------------------------
    // 0. 准备常量与缓冲区
    // -------------------------------------------------------------------------
    const auto& global = GlobalData::getInstance();
    real_t gamma = global.gamma;
    real_t efix = config_->scheme.entropy_fix;
    
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;

    // 计算最大维度用于分配一维缓存
    int nmax = std::max({ni, nj, nk});
    // 缓冲区大小：覆盖 -2 到 nmax+3 (共 nmax + 6)
    int line_size = nmax + 6; 
    int offset = 2; // 用于将 -2 映射到 0

    // --- 分配临时缓冲区 (Line Buffers) ---
    // 1. 原始变量线缓存 (对应 q_line(..., 1))
    // 内容: r, u, v, w, p, gamma
    // 大小: [line_size][6]
    std::vector<std::array<real_t, 6>> q_line_prim(line_size);

    // 2. 守恒变量线缓存 (对应 q_line(..., 2))
    // 内容: q0, q1, q2, q3, q4, T
    // 大小: [line_size][6]
    std::vector<std::array<real_t, 6>> q_line_cons(line_size);

    // 3. 几何度量线缓存 (对应 trxyz)
    // 内容: kx, ky, kz, kt, vol
    // 大小: [nmax][5] (Fortran只用到物理区 1:n)
    // 这里我们稍微开大一点防止越界，使用 nmax
    std::vector<std::array<real_t, 5>> trxyz(nmax);

    // 4. 通量/残差增量缓存 (对应 fc)
    // 内容: 5 个守恒分量
    // 大小: [nmax][5]
    std::vector<std::array<real_t, 5>> fc(nmax);

    // -------------------------------------------------------------------------
    // 1. I-Direction Sweep (I 方向扫描)
    // 对应 Fortran: do k=1,nk; do j=1,nj
    // -------------------------------------------------------------------------
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            
            // --- 1.1 提取 I 方向线数据 ---
            // Range: -2 到 ni+3 (包含 Ghost)
            for (int i_idx = -2; i_idx <= ni + 3; ++i_idx) {
                // 映射到 Block 的全局索引 (0-based with ghost)
                // Fortran i=1 对应 C++ ng. 所以 i_idx 对应 i_idx - 1 + ng
                int I = i_idx + ng - 1;
                // 钳位保护 (Clamp)，防止越界访问
                // 虽然逻辑上 -2 对应 ng-3 应该是安全的 (只要 ng>=3)，但加保护是个好习惯
                if (I < 0) I = 0;
                if (I >= ni + 2 * ng) I = ni + 2 * ng - 1;
                
                int J = j + ng;
                int K = k + ng;

                // 填充 Primitive Variables
                q_line_prim[i_idx + offset][0] = b->r(I, J, K);
                q_line_prim[i_idx + offset][1] = b->u(I, J, K);
                q_line_prim[i_idx + offset][2] = b->v(I, J, K);
                q_line_prim[i_idx + offset][3] = b->w(I, J, K);
                q_line_prim[i_idx + offset][4] = b->p(I, J, K);
                q_line_prim[i_idx + offset][5] = gamma; // Gamma

                // 填充 Conservative Variables + Temperature
                for (int m = 0; m < 5; ++m) {
                    q_line_cons[i_idx + offset][m] = b->q(I, J, K, m);
                }
                q_line_cons[i_idx + offset][5] = b->t(I, J, K); // Temperature
            }

            // --- 1.2 提取几何度量 (Case Default: Standard WCNS) ---
            // Range: 1 到 ni (物理区) -> C++ 0 到 ni-1
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;

                trxyz[i][0] = b->kcx(I, J, K);
                trxyz[i][1] = b->kcy(I, J, K);
                trxyz[i][2] = b->kcz(I, J, K);
                trxyz[i][3] = b->kct(I, J, K);
                trxyz[i][4] = b->vol(I, J, K);
            }

            // --- 1.3 计算通量 (待实现) ---
            flux_line_wcns_roe(ni, q_line_prim, q_line_cons, trxyz, efix, fc);
            
            // --- 1.4 更新残差 ---
            // 对应 Fortran: do i=1,ni; dq = dq + fc
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                for (int m = 0; m < 5; ++m) {
                    // TODO: 这里的 fc[i][m] 暂时是 0，等 flux_line 实现后即可
                    b->dq(I, J, K, m) += fc[i][m];
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. J-Direction Sweep (J 方向扫描)
    // 对应 Fortran: do k=1,nk; do i=1,ni
    // -------------------------------------------------------------------------
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            
            // --- 2.1 提取 J 方向线数据 ---
            for (int j_idx = -2; j_idx <= nj + 3; ++j_idx) {
                int J = j_idx + ng - 1;
                if (J < 0) J = 0;
                if (J >= nj + 2 * ng) J = nj + 2 * ng - 1;

                int I = i + ng;
                int K = k + ng;

                q_line_prim[j_idx + offset][0] = b->r(I, J, K);
                q_line_prim[j_idx + offset][1] = b->u(I, J, K);
                q_line_prim[j_idx + offset][2] = b->v(I, J, K);
                q_line_prim[j_idx + offset][3] = b->w(I, J, K);
                q_line_prim[j_idx + offset][4] = b->p(I, J, K);
                q_line_prim[j_idx + offset][5] = gamma;

                for (int m = 0; m < 5; ++m) {
                    q_line_cons[j_idx + offset][m] = b->q(I, J, K, m);
                }
                q_line_cons[j_idx + offset][5] = b->t(I, J, K);
            }

            // --- 2.2 提取几何度量 (使用 etx, ety, ...) ---
            for (int j = 0; j < nj; ++j) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;

                trxyz[j][0] = b->etx(I, J, K);
                trxyz[j][1] = b->ety(I, J, K);
                trxyz[j][2] = b->etz(I, J, K);
                trxyz[j][3] = b->ett(I, J, K);
                trxyz[j][4] = b->vol(I, J, K);
            }

            // --- 2.3 计算通量 (待实现) ---
            flux_line_wcns_roe(nj, q_line_prim, q_line_cons, trxyz, efix, fc);

            // --- 2.4 更新残差 ---
            for (int j = 0; j < nj; ++j) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                for (int m = 0; m < 5; ++m) {
                    b->dq(I, J, K, m) += fc[j][m];
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. K-Direction Sweep (K 方向扫描)
    // 对应 Fortran: do j=1,nj; do i=1,ni
    // -------------------------------------------------------------------------
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {

            // --- 3.1 提取 K 方向线数据 ---
            for (int k_idx = -2; k_idx <= nk + 3; ++k_idx) {
                int K = k_idx + ng - 1;
                if (K < 0) K = 0;
                if (K >= nk + 2 * ng) K = nk + 2 * ng - 1;

                int I = i + ng;
                int J = j + ng;

                q_line_prim[k_idx + offset][0] = b->r(I, J, K);
                q_line_prim[k_idx + offset][1] = b->u(I, J, K);
                q_line_prim[k_idx + offset][2] = b->v(I, J, K);
                q_line_prim[k_idx + offset][3] = b->w(I, J, K);
                q_line_prim[k_idx + offset][4] = b->p(I, J, K);
                q_line_prim[k_idx + offset][5] = gamma;

                for (int m = 0; m < 5; ++m) {
                    q_line_cons[k_idx + offset][m] = b->q(I, J, K, m);
                }
                q_line_cons[k_idx + offset][5] = b->t(I, J, K);
            }

            // --- 3.2 提取几何度量 (使用 ctx, cty, ...) ---
            for (int k = 0; k < nk; ++k) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;

                trxyz[k][0] = b->ctx(I, J, K);
                trxyz[k][1] = b->cty(I, J, K);
                trxyz[k][2] = b->ctz(I, J, K);
                trxyz[k][3] = b->ctt(I, J, K);
                trxyz[k][4] = b->vol(I, J, K);
            }

            // --- 3.3 计算通量 (待实现) ---
            flux_line_wcns_roe(nk, q_line_prim, q_line_cons, trxyz, efix, fc);

            // --- 3.4 更新残差 ---
            for (int k = 0; k < nk; ++k) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                for (int m = 0; m < 5; ++m) {
                    b->dq(I, J, K, m) += fc[k][m];
                }
            }
        }
    }
}

// =========================================================================
// 驱动函数：计算一条线上的通量导数 (WCNS + Roe)
// 1. 调用 value_half_node_static 计算几何
// 2. 调用 flux_dxyz_static 计算通量导数
// 3. 严格复刻 Fortran 边界迭代逻辑 (Backup 1..4)
// =========================================================================
void NsKernel::flux_line_wcns_roe(
    int ni, 
    const std::vector<std::array<real_t, 6>>& q_line_prim, 
    const std::vector<std::array<real_t, 6>>& q_line_cons, 
    const std::vector<std::array<real_t, 5>>& trxyz,       
    real_t efix,  
    std::vector<std::array<real_t, 5>>& fc                 
) {
    const int nl = 5; 
    const real_t small = 1.0e-38;
    const real_t rmin_limit = 1.0e-20; 
    const real_t pmin_limit = 1.0e-20; 
    const real_t EPS = 1.0e-6;

    const real_t CL1 = 1.0/16.0, CL2 = 10.0/16.0, CL3 = 5.0/16.0;
    const real_t CR1 = 5.0/16.0, CR2 = 10.0/16.0, CR3 = 1.0/16.0;

    const int q_offset = 2; // q(-2) maps to index 0

    // -------------------------------------------------------------------------
    // 1. 数据准备：扁平化 (SoA) 以适配子函数
    // -------------------------------------------------------------------------
    int len_q = ni + 6; 
    int len_face = ni + 1; 

    // Q SoA: [m][i] via vector of vectors provided to main loop logic
    // We keep q_soa as vector of vectors for ease of sliding window access in WCNS loop
    std::vector<std::vector<real_t>> q_soa(6, std::vector<real_t>(len_q));
    for (int i = 0; i < len_q; ++i) {
        for (int m = 0; m < 6; ++m) q_soa[m][i] = q_line_prim[i][m];
    }

    // Geometry SoA Flat: [m * ni + i] for value_half_node
    std::vector<real_t> trxyz_flat(5 * ni);
    for (int m = 0; m < 5; ++m) {
        for (int i = 0; i < ni; ++i) {
            trxyz_flat[m * ni + i] = trxyz[i][m];
        }
    }

    // Output Flat Buffers
    std::vector<real_t> nx_flat(5 * len_face); // Output of value_half_node
    std::vector<real_t> f_flat(nl * len_face); // Input for flux_dxyz
    std::vector<real_t> df_flat(nl * ni);      // Output of flux_dxyz

    // -------------------------------------------------------------------------
    // 2. 调用几何插值子函数
    // -------------------------------------------------------------------------
    value_half_node_static(5, ni, trxyz_flat, nx_flat);

    // -------------------------------------------------------------------------
    // 3. WCNS 重构主循环 (手动计算 QL/QR)
    // -------------------------------------------------------------------------
    std::vector<real_t> s1(ni+6), s2(ni+6), s3(ni+6);
    std::vector<real_t> g1(ni+6), g2(ni+6), g3(ni+6);
    std::vector<real_t> bl(3), br(3), wl(3), wr(3);
    
    std::vector<std::vector<real_t>> qwl(nl, std::vector<real_t>(len_face));
    std::vector<std::vector<real_t>> qwr(nl, std::vector<real_t>(len_face));

    int ist = 0, ied = ni;
    if (q_soa[0][0] < small) ist = 2;
    if (q_soa[0][ni+5] < small) ied = ni - 2; 

    // Buffer weights
    std::vector<std::vector<std::array<real_t, 3>>> buffer_wl(nl, std::vector<std::array<real_t, 3>>(len_q));
    std::vector<std::vector<std::array<real_t, 3>>> buffer_wr(nl, std::vector<std::array<real_t, 3>>(len_q));
    std::vector<std::vector<std::array<real_t, 3>>> buffer_s(nl, std::vector<std::array<real_t, 3>>(len_q));
    std::vector<std::vector<std::array<real_t, 3>>> buffer_g(nl, std::vector<std::array<real_t, 3>>(len_q));

    // Pass 1: Weights
    for (int m = 0; m < nl; ++m) {
        const auto& q = q_soa[m];
        int idx_pre = (ist - 1) + q_offset;
        s2[idx_pre] = q[idx_pre-1] - 2.0*q[idx_pre] + q[idx_pre+1];
        s3[idx_pre] = q[idx_pre] - 2.0*q[idx_pre+1] + q[idx_pre+2];

        for (int i = ist; i <= ied + 1; ++i) {
            int idx = i + q_offset;
            g1[idx] = 0.5 * (q[idx-2] - 4.0*q[idx-1] + 3.0*q[idx]);
            g2[idx] = 0.5 * (q[idx+1] - q[idx-1]);
            g3[idx] = 0.5 * (-3.0*q[idx] + 4.0*q[idx+1] - q[idx+2]);
            s1[idx] = s2[idx-1]; s2[idx] = s3[idx-1]; s3[idx] = q[idx] - 2.0*q[idx+1] + q[idx+2];

            buffer_g[m][idx][0] = g1[idx]; buffer_g[m][idx][1] = g2[idx]; buffer_g[m][idx][2] = g3[idx];
            buffer_s[m][idx][0] = s1[idx]; buffer_s[m][idx][1] = s2[idx]; buffer_s[m][idx][2] = s3[idx];

            auto calc_and_norm = [&](const real_t* g, const real_t* s, const real_t* C, std::array<real_t, 3>& w_out) {
                real_t b[3], IS_sum = 0;
                for(int k=0; k<3; ++k) {
                    real_t IS = g[k]*g[k] + s[k]*s[k];
                    b[k] = C[k] / ((EPS+IS)*(EPS+IS));
                    IS_sum += b[k];
                }
                for(int k=0; k<3; ++k) w_out[k] = b[k] / IS_sum;
            };
            const real_t g_arr[3] = {g1[idx], g2[idx], g3[idx]};
            const real_t s_arr[3] = {s1[idx], s2[idx], s3[idx]};
            const real_t CL[3] = {CL1, CL2, CL3}, CR[3] = {CR1, CR2, CR3};
            calc_and_norm(g_arr, s_arr, CL, buffer_wl[m][idx]);
            calc_and_norm(g_arr, s_arr, CR, buffer_wr[m][idx]);
        }
    }

    // Pass 2: QL/QR
    for (int i = ist; i <= ied; ++i) {
        int idx = i + q_offset;
        int idx1 = idx + 1; 
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            const auto& wl_vec = buffer_wl[m][idx];
            const auto& s_vec  = buffer_s[m][idx];
            const auto& g_vec  = buffer_g[m][idx];
            const auto& wr_vec = buffer_wr[m][idx1];
            const auto& s1_vec = buffer_s[m][idx1];
            const auto& g1_vec = buffer_g[m][idx1];

            qwl[m][i] = q[idx] + 0.125 * (wl_vec[0]*(s_vec[0] + 4.0*g_vec[0]) + wl_vec[1]*(s_vec[1] + 4.0*g_vec[1]) + wl_vec[2]*(s_vec[2] + 4.0*g_vec[2]));
            qwr[m][i] = q[idx1] + 0.125 * (wr_vec[0]*(s1_vec[0] - 4.0*g1_vec[0]) + wr_vec[1]*(s1_vec[1] - 4.0*g1_vec[1]) + wr_vec[2]*(s1_vec[2] - 4.0*g1_vec[2]));
        }
    }

    // 4. Boundary Reduction
    if (ist > 1) { 
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            qwl[m][0] = (35.0*q[3] - 35.0*q[4] + 21.0*q[5] - 5.0*q[6]) / 16.0;
            qwl[m][1] = ( 5.0*q[3] + 15.0*q[4] -  5.0*q[5] +     q[6]) / 16.0;
            qwl[m][2] = (    -q[3] +  9.0*q[4] +  9.0*q[5] -     q[6]) / 16.0;
            qwr[m][0] = qwl[m][0]; qwr[m][1] = qwl[m][1];
        }
    }

    if (ied < ni) { 
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            int idx_ni = ni + 2;
            qwr[m][ni]   = (35.0*q[idx_ni] - 35.0*q[idx_ni-1] + 21.0*q[idx_ni-2] - 5.0*q[idx_ni-3]) / 16.0;
            qwr[m][ni-1] = ( 5.0*q[idx_ni] + 15.0*q[idx_ni-1] -  5.0*q[idx_ni-2] +     q[idx_ni-3]) / 16.0;
            qwr[m][ni-2] = (    -q[idx_ni] +  9.0*q[idx_ni-1] +  9.0*q[idx_ni-2] -     q[idx_ni-3]) / 16.0;
            qwl[m][ni]   = qwr[m][ni]; qwl[m][ni-1] = qwr[m][ni-1];
        }
    }

    // -------------------------------------------------------------------------
    // 5. Flux Calculation (Main Pass)
    // -------------------------------------------------------------------------
    auto calc_flux_to_flat = [&](int i_start, int i_end) {
        real_t ql_tmp[5], qr_tmp[5], f_tmp[5];
        for (int i = i_start; i <= i_end; ++i) {
            for (int m = 0; m < nl; ++m) {
                ql_tmp[m] = qwl[m][i];
                qr_tmp[m] = qwr[m][i];
            }
            // 物理限制检查: max(i,1)+2 对应物理区起始
            if (ql_tmp[0] <= rmin_limit || ql_tmp[4] <= pmin_limit) {
                for (int m = 0; m < nl; ++m) ql_tmp[m] = q_soa[m][std::max(i, 1) + 2]; 
            }
            if (qr_tmp[0] <= rmin_limit || qr_tmp[4] <= pmin_limit) {
                for (int m = 0; m < nl; ++m) qr_tmp[m] = q_soa[m][std::min(i+1, ni) + 2]; 
            }

            // 从 nx_flat (SoA) 中读取
            // nx_flat structure: [dim * len_face + i]
            real_t nx = nx_flat[0 * len_face + i];
            real_t ny = nx_flat[1 * len_face + i];
            real_t nz = nx_flat[2 * len_face + i];
            // 忽略 nt (dim=3)

            real_t gamaeq = q_soa[5][i + q_offset];

            // 调用不带 nt 的 kernel
            flux_roe_kernel(ql_tmp, qr_tmp, nx, ny, nz, gamaeq, efix, f_tmp);

            // 写入 f_flat (SoA)
            for (int m = 0; m < nl; ++m) f_flat[m * len_face + i] = f_tmp[m];
        }
    };

    calc_flux_to_flat(0, ni);

    // -------------------------------------------------------------------------
    // 6. 调用通量导数子函数
    // -------------------------------------------------------------------------
    flux_dxyz_static(nl, ni, f_flat, df_flat);

    // -------------------------------------------------------------------------
    // 7. 左边界迭代 (Left Boundary) - 手动修正 df
    // -------------------------------------------------------------------------
    // 定义存取 f_soa 的宏，方便写公式
#define F_SOA(m, i) (f_flat[(m) * len_face + (i)])
#define DF_SOA(m, i) (df_flat[(m) * ni + (i)])

    if (ist > 1) {
        std::vector<std::vector<real_t>> u_l_old(nl, std::vector<real_t>(8)), u_r_old(nl, std::vector<real_t>(8));
        
        // 备份 1..4
        for (int i = 1; i <= 4; ++i) { 
             for (int m = 0; m < nl; ++m) { 
                 u_l_old[m][i-1] = qwl[m][i]; 
                 u_r_old[m][i-1] = qwr[m][i]; 
             }
        }

        // Iteration 1
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            qwl[m][0] = (3.0*q[3] - q[4])/2.0; qwr[m][0] = qwl[m][0];
            qwl[m][1] = (19.0*q[3] + 25.0*q[4] - 2.0*q[5])/42.0; qwr[m][1] = qwl[m][1];
            qwl[m][2] = (4.0*q[3] + 3.0*q[4] + 9.0*q[5] + 2.0*q[6])/18.0; qwr[m][2] = qwl[m][2];
            qwl[m][3] = (-2.0*q[3] + 3.0*q[4] + 3.0*q[5] + 2.0*q[6])/6.0; qwr[m][3] = qwl[m][3];
        }
        calc_flux_to_flat(0, 3);
        // Explicit DF update
        for (int m = 0; m < nl; ++m) DF_SOA(m, 0) = (-23.0*F_SOA(m, 0) + 21.0*F_SOA(m, 1) + 3.0*F_SOA(m, 2) - F_SOA(m, 3))/24.0;

        // Iteration 2
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            qwl[m][1] = (3.0*q[3] + 6.0*q[4] - q[5])/8.0; qwr[m][1] = qwl[m][1];
            qwl[m][2] = (-30.0*q[3] + 145.0*q[4] + 29.0*q[5] - 8.0*q[6])/136.0;
            qwr[m][2] = (2.0*q[3] + 49.0*q[4] + 125.0*q[5] - 40.0*q[6])/136.0;
            qwl[m][3] = (-q[4] + 6.0*q[5] + 3.0*q[6])/8.0; qwr[m][3] = qwl[m][3];
            qwl[m][4] = (q[4] + 2.0*q[5] + 5.0*q[6])/8.0; qwr[m][4] = qwl[m][4];
            qwl[m][5] = (q[4] + q[5] + 6.0*q[6])/8.0; qwr[m][5] = qwl[m][5];
        }
        calc_flux_to_flat(1, 5);
        for (int m = 0; m < nl; ++m) DF_SOA(m, 1) = (-22.0*F_SOA(m, 1) + 17.0*F_SOA(m, 2) + 9.0*F_SOA(m, 3) - 5.0*F_SOA(m, 4) + F_SOA(m, 5))/24.0;

        // Iteration 3
        for (int m = 0; m < nl; ++m) {
            qwr[m][1] = u_r_old[m][0]; // restore old 1
            qwl[m][2] = (-29.0*q_soa[m][3] + 170.0*q_soa[m][4] + 63.0*q_soa[m][5] + 12.0*q_soa[m][6])/216.0;
            qwr[m][2] = u_r_old[m][1]; 
            qwr[m][3] = u_r_old[m][2]; 
            qwr[m][4] = u_r_old[m][3]; 
        }
        calc_flux_to_flat(1, 4);
        for (int m = 0; m < nl; ++m) DF_SOA(m, 2) = (F_SOA(m, 1) - 27.0*F_SOA(m, 2) + 27.0*F_SOA(m, 3) - F_SOA(m, 4))/24.0;
    }

    // -------------------------------------------------------------------------
    // 8. 右边界迭代 (Right Boundary)
    // -------------------------------------------------------------------------
    if (ied < ni) {
        std::vector<std::vector<real_t>> u_l_old(nl, std::vector<real_t>(8)), u_r_old(nl, std::vector<real_t>(8));
        for (int i = 1; i <= 4; ++i) {
            int i1 = ni - i; 
            for (int m = 0; m < nl; ++m) { u_l_old[m][9-i-1] = qwl[m][i1]; u_r_old[m][9-i-1] = qwr[m][i1]; }
        }
        int idx_n = ni + 2; 

        // Iteration 1
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            qwr[m][ni] = (3.0*q[idx_n] - q[idx_n-1])/2.0; qwl[m][ni] = qwr[m][ni];
            qwr[m][ni-1] = (19.0*q[idx_n] + 25.0*q[idx_n-1] - 2.0*q[idx_n-2])/42.0; qwl[m][ni-1] = qwr[m][ni-1];
            qwr[m][ni-2] = (4.0*q[idx_n] + 3.0*q[idx_n-1] + 9.0*q[idx_n-2] + 2.0*q[idx_n-3])/18.0; qwl[m][ni-2] = qwr[m][ni-2];
            qwr[m][ni-3] = (-2.0*q[idx_n] + 3.0*q[idx_n-1] + 3.0*q[idx_n-2] + 2.0*q[idx_n-3])/6.0; qwl[m][ni-3] = qwr[m][ni-3];
        }
        calc_flux_to_flat(ni-3, ni);
        for (int m = 0; m < nl; ++m) DF_SOA(m, ni-1) = -(-23.0*F_SOA(m, ni) + 21.0*F_SOA(m, ni-1) + 3.0*F_SOA(m, ni-2) - F_SOA(m, ni-3))/24.0;

        // Iteration 2
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            qwr[m][ni-1] = (3.0*q[idx_n] + 6.0*q[idx_n-1] - q[idx_n-2])/8.0; qwl[m][ni-1] = qwr[m][ni-1];
            qwr[m][ni-2] = (-30.0*q[idx_n] + 145.0*q[idx_n-1] + 29.0*q[idx_n-2] - 8.0*q[idx_n-3])/136.0;
            qwl[m][ni-2] = (2.0*q[idx_n] + 49.0*q[idx_n-1] + 125.0*q[idx_n-2] - 40.0*q[idx_n-3])/136.0;
            qwr[m][ni-3] = (-q[idx_n-1] + 6.0*q[idx_n-2] + 3.0*q[idx_n-3])/8.0; qwl[m][ni-3] = qwr[m][ni-3];
            qwr[m][ni-4] = (q[idx_n-1] + 2.0*q[idx_n-2] + 5.0*q[idx_n-3])/8.0; qwl[m][ni-4] = qwr[m][ni-4];
            qwr[m][ni-5] = (q[idx_n-1] + q[idx_n-2] + 6.0*q[idx_n-3])/8.0; qwl[m][ni-5] = qwr[m][ni-5];
        }
        calc_flux_to_flat(ni-5, ni-1);
        for (int m = 0; m < nl; ++m) DF_SOA(m, ni-2) = -(-22.0*F_SOA(m, ni-1) + 17.0*F_SOA(m, ni-2) + 9.0*F_SOA(m, ni-3) - 5.0*F_SOA(m, ni-4) + F_SOA(m, ni-5))/24.0;

        // Iteration 3
        for (int m = 0; m < nl; ++m) {
            const auto& q = q_soa[m];
            qwl[m][ni-1] = u_l_old[m][8-1];
            qwr[m][ni-2] = (-29.0*q[idx_n] + 170.0*q[idx_n-1] + 63.0*q[idx_n-2] + 12.0*q[idx_n-3])/216.0;
            qwl[m][ni-2] = u_l_old[m][7-1]; qwl[m][ni-3] = u_l_old[m][6-1]; qwl[m][ni-4] = u_l_old[m][5-1];
        }
        calc_flux_to_flat(ni-4, ni-1);
        for (int m = 0; m < nl; ++m) DF_SOA(m, ni-3) = -(F_SOA(m, ni-1) - 27.0*F_SOA(m, ni-2) + 27.0*F_SOA(m, ni-3) - F_SOA(m, ni-4))/24.0;
    }

    // -------------------------------------------------------------------------
    // 9. 回写 (SoA Flat -> fc AoS)
    // -------------------------------------------------------------------------
    for (int i = 0; i < ni; ++i) {
        for (int m = 0; m < nl; ++m) {
            fc[i][m] = DF_SOA(m, i);
        }
    }

#undef F_SOA
#undef DF_SOA
}

// =========================================================================
// 核心计算：Roe 通量格式 (100% 复刻 src-fortran/flux.f90)
// =========================================================================
void NsKernel::flux_roe_kernel(
    const real_t* ql_prim, const real_t* qr_prim, // 输入：左右原始变量 [rho, u, v, w, p]
    real_t nx, real_t ny, real_t nz,              // 输入：界面法向量 (包含面积)
    real_t gamma, real_t efix,                    // 输入：气体参数与熵修正系数
    real_t* f_interface                           // 输出：界面通量 [5]
) {
    // -------------------------------------------------------------------------
    // 0. 局部变量定义 (保持与 Fortran 一致的命名风格)
    // -------------------------------------------------------------------------
    real_t ql[5], qr[5]; // 本地副本，用于支持原地修改逻辑
    real_t dq[5];
    real_t fl[5], fr[5], df[5], qroe[5];
    
    // 中间变量
    real_t l1, l4, l5, rm, um, vm, wm, cm, hm;
    real_t l12, l42, l52;
    real_t cta, cgm, cgm1, rrl, rrl_1, ccgm;
    real_t c2, c, dcta, cta1, gama1, gama2;
    real_t deti, det2, rodctac, rodcta, dp_c2;
    real_t a1, a2, a3, a4, a5, a6, a7, a8;

    // 常量
    const real_t sml_sss = 1.0e-30;
    
    // 初始化本地副本 (因为 Fortran 代码会修改输入变量)
    for(int m = 0; m < 5; ++m) {
        ql[m] = ql_prim[m];
        qr[m] = qr_prim[m];
    }

    // -------------------------------------------------------------------------
    // 1. 计算左右两侧的物理通量 (to calcualte fluxes of right and left sides)
    //    注意：此时 ql, qr 还是原始变量 [rho, u, v, w, p]
    // -------------------------------------------------------------------------
    compute_euler_flux(ql, nx, ny, nz, gamma, fl);
    compute_euler_flux(qr, nx, ny, nz, gamma, fr);

    // -------------------------------------------------------------------------
    // 2. 计算变量差分 (the difference of variables between right and left sides)
    //    dq[4] 此处为 P_R - P_L
    // -------------------------------------------------------------------------
    for (int m = 0; m < 5; ++m) {
        dq[m] = qr[m] - ql[m];
    }

    // -------------------------------------------------------------------------
    // 3. 计算总焓 (to calculate H at left and right sides)
    //    注意：这里将 ql[4], qr[4] 从 压力 修改为 总焓 H
    // -------------------------------------------------------------------------
    gama1 = gamma - 1.0;
    gama2 = gamma / gama1;

    // ql(5)=gama2*ql(5)/ql(1) + 0.5*(ql(2)**2+ql(3)**2+ql(4)**2)
    ql[4] = gama2 * ql[4] / ql[0] + 0.5 * (ql[1]*ql[1] + ql[2]*ql[2] + ql[3]*ql[3]);
    qr[4] = gama2 * qr[4] / qr[0] + 0.5 * (qr[1]*qr[1] + qr[2]*qr[2] + qr[3]*qr[3]);

    // -------------------------------------------------------------------------
    // 4. 计算 Roe 平均量 (to calculate density of Roe average)
    // -------------------------------------------------------------------------
    qroe[0] = std::sqrt(ql[0] * qr[0]); // Roe Density

    // to calculate velocity and H of Roe average
    rrl = std::sqrt(qr[0] / ql[0]);
    rrl_1 = rrl + 1.0;

    for (int m = 1; m < 5; ++m) {
        qroe[m] = (ql[m] + qr[m] * rrl) / rrl_1;
    }

    rm = qroe[0];
    um = qroe[1];
    vm = qroe[2];
    wm = qroe[3];
    hm = qroe[4]; // Roe Enthalpy

    // -------------------------------------------------------------------------
    // 5. 计算声速 (to calculate the speed of sound)
    // -------------------------------------------------------------------------
    c2 = (hm - 0.5 * (um*um + vm*vm + wm*wm)) * gama1;
    
    if (c2 <= 0.0) {
        // write(*,*)'sqrt(c2) is a math error...'
        c2 = 1.0e-12; // 保护逻辑
    }
    c = std::sqrt(c2);

    // -------------------------------------------------------------------------
    // 6. 几何处理 (Geometry processing)
    // -------------------------------------------------------------------------
    // cta: 逆变速度 (Contravariant velocity)
    // 注意：这里的 nx, ny, nz 还是面积加权法向量
    cta = um * nx + vm * ny + wm * nz;

    // cgm: 面积模长
    cgm = std::max(std::sqrt(nx*nx + ny*ny + nz*nz), sml_sss);
    cgm1 = 1.0 / cgm;

    // 归一化法向量 (Normalize normals)
    // Fortran: nx = nx*cgm1 ...
    nx = nx * cgm1;
    ny = ny * cgm1;
    nz = nz * cgm1;

    ccgm = c * cgm;

    l1 = std::abs(cta - ccgm);
    l4 = std::abs(cta);
    l5 = std::abs(cta + ccgm);

    // -------------------------------------------------------------------------
    // 7. 熵修正 (Harten's entropy modification)
    // -------------------------------------------------------------------------
    l12 = l1 * l1;
    l42 = l4 * l4;
    l52 = l5 * l5;

    deti = efix * cgm;
    det2 = deti * deti;

    // Fortran 代码中无条件执行 sqrt
    l1 = std::sqrt(l12 + det2);
    l4 = std::sqrt(l42 + det2);
    l5 = std::sqrt(l52 + det2);

    // -------------------------------------------------------------------------
    // 8. 耗散项计算
    // -------------------------------------------------------------------------
    // dcta: 速度差在法向上的投影 (使用 dq[1], dq[2], dq[3] 即 du, dv, dw)
    // 注意：此时 nx, ny, nz 已经是归一化的单位向量
    dcta = dq[1] * nx + dq[2] * ny + dq[3] * nz;

    cta1 = cta * cgm1; // Normalized Contravariant Velocity (Un)

    rodcta = qroe[0] * dcta; // rho * dUn
    dp_c2  = dq[4] / c2;     // dp / c^2 (dq[4] 是 dp)

    rodctac = rodcta / c;

    // 特征波强度系数
    a1 = l4 * (dq[0] - dp_c2);
    a2 = l5 * (dp_c2 + rodctac) * 0.5;
    a3 = l1 * (dp_c2 - rodctac) * 0.5;
    
    a4 = a1 + a2 + a3;
    a5 = c * (a2 - a3);
    
    // 剪切波部分
    a6 = l4 * (rm * dq[1] - nx * rodcta);
    a7 = l4 * (rm * dq[2] - ny * rodcta);
    a8 = l4 * (rm * dq[3] - nz * rodcta);

    // 组装耗散通量 df
    df[0] = a4;
    df[1] = um*a4 + nx*a5 + a6;
    df[2] = vm*a4 + ny*a5 + a7;
    df[3] = wm*a4 + nz*a5 + a8;
    df[4] = hm*a4 + cta1*a5 + um*a6 + vm*a7 + wm*a8 - c2*a1/gama1;

    // -------------------------------------------------------------------------
    // 9. 最终界面通量 (Final Flux)
    // -------------------------------------------------------------------------
    for (int m = 0; m < 5; ++m) {
        f_interface[m] = 0.5 * (fr[m] + fl[m] - df[m]);
    }
}

}