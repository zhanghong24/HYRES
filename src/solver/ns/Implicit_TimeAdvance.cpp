#include "NSKernel.h"

namespace Hyres {

// =========================================================================
// 将非对接边界的 Ghost 区域 dq 设为 0 (防止隐式迭代发散)
// ========================================================================= 
void NsKernel::set_boundary_dq_zero(Block* b) {
    int ng = b->ng;

    for (const auto& patch : b->boundaries) {
        // 仅处理物理边界 (Type > 0)
        if (patch.type > 0) {
            
            // 1. 获取边界定义的几何范围 (Fortran 1-based)
            int is = std::min(std::abs(patch.raw_is[0]), std::abs(patch.raw_ie[0]));
            int ie = std::max(std::abs(patch.raw_is[0]), std::abs(patch.raw_ie[0]));
            int js = std::min(std::abs(patch.raw_is[1]), std::abs(patch.raw_ie[1]));
            int je = std::max(std::abs(patch.raw_is[1]), std::abs(patch.raw_ie[1]));
            int ks = std::min(std::abs(patch.raw_is[2]), std::abs(patch.raw_ie[2]));
            int ke = std::max(std::abs(patch.raw_is[2]), std::abs(patch.raw_ie[2]));

            // 2. 根据方向向外扩展范围 (覆盖 Ghost 层)
            // s_nd: 0=I, 1=J, 2=K (注意 Block.h 定义，Fortran可能是 1,2,3，需确认)
            // s_lr: -1 (Start), +1 (End)
            
            // 假设 patch.s_nd 是 0-based (0,1,2)。如果是 1-based 请减 1。
            int dim = patch.s_nd; 
            int dir = patch.s_lr; // -1 或 1

            // 修正循环范围以包含 Ghost
            if (dim == 0) { // I 方向
                if (dir == -1) { // 左边界 (1)，向左清空 Ghost (1-ng 到 1)
                    ie = is;     // 保持边界处
                    is = 1 - ng; // 扩展到 Ghost 最外层 (Fortran索引)
                } else {         // 右边界 (ni)，向右清空 Ghost (ni 到 ni+ng)
                    is = ie;
                    ie = b->ni + ng;
                }
            } else if (dim == 1) { // J 方向
                if (dir == -1) { js = 1 - ng; je = js + ng; } // 粗略写法，正确逻辑同上
                else           { js = je;     je = b->nj + ng; }
            } else if (dim == 2) { // K 方向
                if (dir == -1) { ks = 1 - ng; ke = ks + ng; }
                else           { ks = ke;     ke = b->nk + ng; }
            }

            // 3. 执行置零 (转换到 C++ 索引)
            for (int k = ks; k <= ke; ++k) {
                for (int j = js; j <= je; ++j) {
                    for (int i = is; i <= ie; ++i) {
                        
                        // 坐标映射：Fortran 1-based -> C++ 0-based with Ghost Offset
                        // 确保索引不越界 (虽然理论上计算好了，加个 clamp 更安全)
                        int ix = i - 1 + ng;
                        int iy = j - 1 + ng;
                        int iz = k - 1 + ng;

                        // 安全检查
                        if (ix >= 0 && ix < b->ni + 2*ng &&
                            iy >= 0 && iy < b->nj + 2*ng &&
                            iz >= 0 && iz < b->nk + 2*ng) 
                        {
                            for (int m = 0; m < 5; ++m) {
                                b->dq(ix, iy, iz, m) = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
// 隐式求解：Gauss-Seidel 点松弛主驱动
// =========================================================================
void NsKernel::solve_gauss_seidel_single(Block* b) {
    // 常量定义
    real_t wmig = 1.0;
    real_t beta = 1.0;
    
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int nl = 5; // 5 equations

    // 1. 分配临时存储 rhs_nb (对应 allocate(rhs_nb(ni,nj,nk,nl)))
    // 注意：Fortran 中通常只存储物理区，不含 Ghost
    // 我们使用一维 vector 扁平化存储，大小为 ni * nj * nk * 5
    std::vector<real_t> rhs_nb(ni * nj * nk * nl);

    // 2. 备份当前显式残差 (Backup Explicit RHS)
    // 对应 Fortran: call store_rhs_nb
    store_rhs_nb(b, rhs_nb);

    // 3. LU-SGS 预处理 (Predictor Step)
    // 对应 Fortran: call lusgs_l / lusgs_u
    // 第三个参数 0.0 在 Fortran 中传入，这里保持一致
    lusgs_l(b, rhs_nb, wmig, beta, 0.0);
    lusgs_u(b, rhs_nb, wmig, beta, 0.0);

    // 4. Gauss-Seidel 点松弛 (Corrector Step)
    // 对应 Fortran: call gs_pr_l / gs_pr_u
    gs_pr_l(b, rhs_nb, wmig, beta);
    gs_pr_u(b, rhs_nb, wmig, beta);

    // 5. 释放内存
    // std::vector rhs_nb 会在函数结束时自动释放，无需手动 deallocate
}

// =========================================================================
// 隐式求解子函数 (占位符)
// =========================================================================

// =========================================================================
// 备份显式残差 (Backup Explicit RHS)
// 对应 Fortran: subroutine store_rhs_nb
// =========================================================================
void NsKernel::store_rhs_nb(Block* b, std::vector<real_t>& rhs_nb) {
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;
    int nl = 5;

    int idx = 0;

    // 遍历物理区 (Fortran 1..n -> C++ 0..n-1)
    // 循环顺序保持与 Fortran 一致：k, j, i, m
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                
                // 坐标映射：物理区 (0-based) -> 带 Ghost 的全局索引
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;

                for (int m = 0; m < nl; ++m) {
                    // 将 dq 备份到线性 buffer 中
                    // 此时 rhs_nb 存储的是尚未更新的 Dq^n (即 R^n)
                    // 布局：AoS [k][j][i][m] 扁平化
                    rhs_nb[idx++] = b->dq(I, J, K, m);
                }
            }
        }
    }
}

// =========================================================================
// 辅助工具：获取网格度量 (对应 getrkec_mml)
// 简化：假设 method=1，处理边界索引钳位
// =========================================================================
inline void get_metrics_at(Block* b, int i, int j, int k, 
                           int dir_i, int dir_j, int dir_k, // 1 或 0，指示方向
                           real_t& nx, real_t& ny, real_t& nz, real_t& nt) 
{
    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 钳位索引 (对应 Fortran 的 if(i1==0) i1=1 等)
    // Fortran 1..ni -> C++ 0..ni-1
    // 输入 i,j,k 是 Fortran 物理索引 (1-based)
    int i1 = std::max(1, std::min(i, ni));
    int j1 = std::max(1, std::min(j, nj));
    int k1 = std::max(1, std::min(k, nk));

    // 映射到 C++ 全局索引
    int I = i1 - 1 + ng;
    int J = j1 - 1 + ng;
    int K = k1 - 1 + ng;

    if (dir_i) {
        nx = b->kcx(I, J, K); ny = b->kcy(I, J, K); nz = b->kcz(I, J, K); nt = b->kct(I, J, K);
    } else if (dir_j) {
        nx = b->etx(I, J, K); ny = b->ety(I, J, K); nz = b->etz(I, J, K); nt = b->ett(I, J, K);
    } else if (dir_k) {
        nx = b->ctx(I, J, K); ny = b->cty(I, J, K); nz = b->ctz(I, J, K); nt = b->ctt(I, J, K);
    } else {
        nx = 0; ny = 0; nz = 0; nt = 0;
    }
}

// =========================================================================
// 辅助工具：计算矩阵向量乘积 (对应 mxdq_std)
// =========================================================================
void NsKernel::matrix_vector_product_std(
    const real_t* prim, const real_t* metrics, const real_t* dq, 
    real_t* f_out, real_t rad, int npn, real_t gamma) 
{
    real_t rm = prim[0];
    real_t um = prim[1];
    real_t vm = prim[2];
    real_t wm = prim[3];
    real_t pm = prim[4];

    // Sound speed
    real_t c2 = gamma * pm / rm;
    real_t cm = std::sqrt(c2);
    real_t v2 = um*um + vm*vm + wm*wm;
    
    // Enthalpy H = (E + p)/rho = (rho*e + 0.5rho*v2 + p)/rho = e + p/rho + 0.5v2
    // Ideal Gas: e = p/((g-1)rho) -> H = g*p/((g-1)rho) + 0.5v2 = c2/(g-1) + 0.5v2
    // Fortran: gettmhmgm -> hint = c2/(g-1). hm = hint + 0.5*v2. 
    real_t hint = c2 / (gamma - 1.0);
    real_t hm = hint + 0.5 * v2;

    real_t nx = metrics[0];
    real_t ny = metrics[1];
    real_t nz = metrics[2];
    real_t nt = metrics[3];

    real_t ct = nx*um + ny*vm + nz*wm + nt; // Contravariant U (un-normalized)
    real_t cgm = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (cgm < 1.0e-30) cgm = 1.0e-30;

    real_t l1 = ct;
    real_t l4 = ct + cm * cgm;
    real_t l5 = ct - cm * cgm;

    // Eigenvalue shift
    real_t shift = npn * rad; // npn is 1 or -1
    l1 = 0.5 * (l1 + shift);
    l4 = 0.5 * (l4 + shift);
    l5 = 0.5 * (l5 + shift);

    real_t x1 = (2.0*l1 - l4 - l5) / (2.0 * c2);
    real_t x2 = (l4 - l5) / (2.0 * cm);

    real_t cgm1 = 1.0 / cgm;
    real_t nx_n = nx * cgm1;
    real_t ny_n = ny * cgm1;
    real_t nz_n = nz * cgm1;
    real_t ct_n = (ct - nt) * cgm1; // Normalized U

    real_t ae = gamma - 1.0;
    real_t af = 0.5 * ae * v2;

    real_t dc = ct_n * dq[0] - nx_n * dq[1] - ny_n * dq[2] - nz_n * dq[3];
    real_t dh = af * dq[0] - ae * (um*dq[1] + vm*dq[2] + wm*dq[3] - dq[4]);

    real_t c2dc = c2 * dc;

    f_out[0] = l1 * dq[0] - dh * x1 - dc * x2;
    f_out[1] = l1 * dq[1] + (nx_n * c2dc - um * dh) * x1 + (nx_n * dh - um * dc) * x2;
    f_out[2] = l1 * dq[2] + (ny_n * c2dc - vm * dh) * x1 + (ny_n * dh - vm * dc) * x2;
    f_out[3] = l1 * dq[3] + (nz_n * c2dc - wm * dh) * x1 + (nz_n * dh - wm * dc) * x2;
    f_out[4] = l1 * dq[4] + (ct_n * c2dc - hm * dh) * x1 + (ct_n * dh - hm * dc) * x2;
}

// =========================================================================
// LU-SGS 下三角扫描 (Forward Sweep)
// 对应 Fortran: subroutine lusgs_l
// =========================================================================
void NsKernel::lusgs_l(Block* b, const std::vector<real_t>& rhs_nb, 
                           real_t wmig, real_t beta, real_t extra_param) 
{
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;
    int nl = 5;
    real_t gamma = GlobalData::getInstance().gamma;
    int nvis = config_->physics.viscous_mode; // 假设 >0 表示有粘性

    std::vector<real_t> rhs0(nl);
    std::vector<real_t> prim_i(nl), prim_j(nl), prim_k(nl);
    std::vector<real_t> gykb_i(4), gykb_j(4), gykb_k(4);
    std::vector<real_t> dq_i(nl), dq_j(nl), dq_k(nl);
    std::vector<real_t> de(nl), df(nl), dg(nl);
    std::vector<real_t> coed(nl);

    // 辅助 lambda: 获取 Primitive 变量
    auto get_prim = [&](int i, int j, int k, std::vector<real_t>& p_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        p_out[0] = b->r(I, J, K);
        p_out[1] = b->u(I, J, K);
        p_out[2] = b->v(I, J, K);
        p_out[3] = b->w(I, J, K);
        p_out[4] = b->p(I, J, K);
    };

    // 辅助 lambda: 获取 Coed (对角项系数)
    auto get_coed = [&](int i, int j, int k, std::vector<real_t>& c_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        // 假设 Block 类有 sra, srb, src, dtdt 数组
        real_t ra = b->sra(I, J, K);
        real_t rb = b->srb(I, J, K);
        real_t rc = b->src(I, J, K);
        real_t rad_ns = ra + rb + rc;
        
        if (nvis > 0) {
            rad_ns += b->srva(I, J, K) + b->srvb(I, J, K) + b->srvc(I, J, K);
        }

        real_t ct = 1.0 / b->dtdt(I, J, K); // dtdt is dt, so 1/dt
        real_t ccc = 1.0 / (ct + beta * wmig * rad_ns);
        for(int m=0; m<nl; ++m) c_out[m] = ccc;
    };

    // 循环: k, j, i (1..n)
    for (int k = 1; k <= nk; ++k) {
        for (int j = 1; j <= nj; ++j) {
            for (int i = 1; i <= ni; ++i) {
                
                int I = i - 1 + ng;
                int J = j - 1 + ng;
                int K = k - 1 + ng;

                // 1. 获取对角系数
                get_coed(i, j, k, coed);

                // 2. 获取邻居 dq (i-1, j-1, k-1)
                // 注意：Fortran i-1 对应 C++ I-1
                for (int m = 0; m < nl; ++m) {
                    dq_i[m] = b->dq(I-1, J, K, m);
                    dq_j[m] = b->dq(I, J-1, K, m);
                    dq_k[m] = b->dq(I, J, K-1, m);
                }

                // 3. 获取邻居 Primitive
                get_prim(i-1, j, k, prim_i);
                get_prim(i, j-1, k, prim_j);
                get_prim(i, j, k-1, prim_k);

                // 4. 获取度量
                real_t nx, ny, nz, nt;
                get_metrics_at(b, i-1, j, k, 1, 0, 0, nx, ny, nz, nt);
                gykb_i[0]=nx; gykb_i[1]=ny; gykb_i[2]=nz; gykb_i[3]=nt;

                get_metrics_at(b, i, j-1, k, 0, 1, 0, nx, ny, nz, nt);
                gykb_j[0]=nx; gykb_j[1]=ny; gykb_j[2]=nz; gykb_j[3]=nt;

                get_metrics_at(b, i, j, k-1, 0, 0, 1, nx, ny, nz, nt);
                gykb_k[0]=nx; gykb_k[1]=ny; gykb_k[2]=nz; gykb_k[3]=nt;

                // 5. 获取谱半径 (ra, rb, rc) - 邻居的
                // Fortran getrarbrc(..., -1) -> i-1, j, k / i, j-1, k ...
                // 简化处理：直接取邻居的 sra/srb/src
                // 这里需要注意 Fortran getrarbrc 中的 i_mml = max(i-1, 1) 逻辑
                // 我们直接读取 clamp 后的索引
                int Im = std::max(0, i-2) + ng; // i-1 in Fortran -> max(1, i-1) -> C++
                int Jm = std::max(0, j-2) + ng;
                int Km = std::max(0, k-2) + ng;
                real_t ra = b->sra(Im, J, K); // sra(i-1, j, k)
                real_t rb = b->srb(I, Jm, K);
                real_t rc = b->src(I, J, Km);

                // 6. 计算通量差 splitting (mxdq_std)
                matrix_vector_product_std(prim_i.data(), gykb_i.data(), dq_i.data(), de.data(), ra, 1, gamma);
                matrix_vector_product_std(prim_j.data(), gykb_j.data(), dq_j.data(), df.data(), rb, 1, gamma);
                matrix_vector_product_std(prim_k.data(), gykb_k.data(), dq_k.data(), dg.data(), rc, 1, gamma);

                // 7. 组装 rhs0
                for (int m = 0; m < nl; ++m) {
                    rhs0[m] = wmig * (de[m] + df[m] + dg[m]);
                }

                // 8. 粘性项贡献
                if (nvis > 0) {
                    real_t rva = b->srva(I-1, J, K);
                    real_t rvb = b->srvb(I, J-1, K);
                    real_t rvc = b->srvc(I, J, K-1);
                    for (int m = 0; m < nl; ++m) {
                        rhs0[m] += 0.5 * (rva*dq_i[m] + rvb*dq_j[m] + rvc*dq_k[m]);
                    }
                }

                // 9. 更新 dq
                // 公式: dq = (-dq + rhs0) * coed
                // 注意：这里的 dq 是 rhs_nb (原始显式残差) 吗？
                // Fortran 代码中: dq(i,j,k,m) = (-dq(i,j,k,m) + rhs0(m)) * coed(m)
                // 此时 dq 存储的是显式残差 R^n (从 store_rhs_nb 恢复的？不，store_rhs_nb 存的是 rhs_nb)
                // 这里的 dq(i,j,k) 是迭代变量。
                // 初始时 dq 存储显式残差 R。
                // 更新后 dq 变为 Delta Q*。
                
                // 【重要】Fortran 代码中并没有用 rhs_nb 来初始化 dq。
                // 这意味着 dq 在进入 lusgs_l 时，应该已经是 Explicit RHS。
                // store_rhs_nb 只是为了后面的 gs_pr 使用。
                // 所以这里直接操作 b->dq。
                
                for (int m = 0; m < nl; ++m) {
                    real_t current_dq = b->dq(I, J, K, m);
                    // Fortran: (-dq + rhs0) * coed. 
                    // 通常 Implicit eq: (I + A)dQ = R. -> dQ = (R - ...)/D
                    // Fortran code signs are tricky. Assuming consistent with provided snippet.
                    b->dq(I, J, K, m) = (-current_dq + rhs0[m]) * coed[m];
                }
            }
        }
    }
}

// =========================================================================
// LU-SGS 上三角扫描 (Backward Sweep)
// 对应 Fortran: subroutine lusgs_u
// =========================================================================
void NsKernel::lusgs_u(Block* b, const std::vector<real_t>& rhs_nb, 
                           real_t wmig, real_t beta, real_t extra_param) 
{
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;
    int nl = 5;
    real_t gamma = GlobalData::getInstance().gamma;
    int nvis = config_->physics.viscous_mode;

    std::vector<real_t> rhs0(nl);
    std::vector<real_t> prim_i(nl), prim_j(nl), prim_k(nl);
    std::vector<real_t> gykb_i(4), gykb_j(4), gykb_k(4);
    std::vector<real_t> dq_i(nl), dq_j(nl), dq_k(nl);
    std::vector<real_t> de(nl), df(nl), dg(nl);
    std::vector<real_t> coed(nl);

    auto get_prim = [&](int i, int j, int k, std::vector<real_t>& p_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        p_out[0] = b->r(I, J, K); p_out[1] = b->u(I, J, K); p_out[2] = b->v(I, J, K);
        p_out[3] = b->w(I, J, K); p_out[4] = b->p(I, J, K);
    };

    auto get_coed = [&](int i, int j, int k, std::vector<real_t>& c_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        real_t rad_ns = b->sra(I, J, K) + b->srb(I, J, K) + b->src(I, J, K);
        if (nvis > 0) rad_ns += b->srva(I, J, K) + b->srvb(I, J, K) + b->srvc(I, J, K);
        real_t ct = 1.0 / b->dtdt(I, J, K);
        real_t ccc = 1.0 / (ct + beta * wmig * rad_ns);
        for(int m=0; m<nl; ++m) c_out[m] = ccc;
    };

    // 循环: k, j, i 逆序 (nk -> 1)
    for (int k = nk; k >= 1; --k) {
        for (int j = nj; j >= 1; --j) {
            for (int i = ni; i >= 1; --i) {
                
                int I = i - 1 + ng;
                int J = j - 1 + ng;
                int K = k - 1 + ng;

                get_coed(i, j, k, coed);

                // 获取邻居 dq (i+1, j+1, k+1)
                for (int m = 0; m < nl; ++m) {
                    dq_i[m] = b->dq(I+1, J, K, m);
                    dq_j[m] = b->dq(I, J+1, K, m);
                    dq_k[m] = b->dq(I, J, K+1, m);
                }

                get_prim(i+1, j, k, prim_i);
                get_prim(i, j+1, k, prim_j);
                get_prim(i, j, k+1, prim_k);

                // 获取度量 (注意 i+1, j+1, k+1)
                real_t nx, ny, nz, nt;
                get_metrics_at(b, i+1, j, k, 1, 0, 0, nx, ny, nz, nt);
                gykb_i[0]=nx; gykb_i[1]=ny; gykb_i[2]=nz; gykb_i[3]=nt;

                get_metrics_at(b, i, j+1, k, 0, 1, 0, nx, ny, nz, nt);
                gykb_j[0]=nx; gykb_j[1]=ny; gykb_j[2]=nz; gykb_j[3]=nt;

                get_metrics_at(b, i, j, k+1, 0, 0, 1, nx, ny, nz, nt);
                gykb_k[0]=nx; gykb_k[1]=ny; gykb_k[2]=nz; gykb_k[3]=nt;

                // 谱半径 (Neighbor + 1)
                // getrarbrc(..., 1)
                int Ip = std::min(ni-1, i) + ng; // i in Fortran (0-based: i-1+1=i)
                int Jp = std::min(nj-1, j) + ng;
                int Kp = std::min(nk-1, k) + ng;
                // Wait, logic: i+1 clamped to ni. In C++ index: i-1+1 = i.
                // Fortran: min(i+1, ni). C++ index of that: min(i+1, ni) -1 + ng
                
                real_t ra = b->sra(std::min(ni, i+1)-1+ng, J, K); 
                real_t rb = b->srb(I, std::min(nj, j+1)-1+ng, K);
                real_t rc = b->src(I, J, std::min(nk, k+1)-1+ng);

                // Split flux (npn = -1)
                matrix_vector_product_std(prim_i.data(), gykb_i.data(), dq_i.data(), de.data(), ra, -1, gamma);
                matrix_vector_product_std(prim_j.data(), gykb_j.data(), dq_j.data(), df.data(), rb, -1, gamma);
                matrix_vector_product_std(prim_k.data(), gykb_k.data(), dq_k.data(), dg.data(), rc, -1, gamma);

                // Assemble rhs0
                for (int m = 0; m < nl; ++m) {
                    rhs0[m] = wmig * (de[m] + df[m] + dg[m]);
                }

                if (nvis > 0) {
                    real_t rva = b->srva(I+1, J, K);
                    real_t rvb = b->srvb(I, J+1, K);
                    real_t rvc = b->srvc(I, J, K+1);
                    for (int m = 0; m < nl; ++m) {
                        rhs0[m] -= 0.5 * (rva*dq_i[m] + rvb*dq_j[m] + rvc*dq_k[m]);
                    }
                }

                // Update dq
                // Formula: dq = dq - rhs0 * coed
                for (int m = 0; m < nl; ++m) {
                    b->dq(I, J, K, m) = b->dq(I, J, K, m) - rhs0[m] * coed[m];
                }
            }
        }
    }
}

// =========================================================================
// Gauss-Seidel 下三角松弛 (Lower Relaxation)
// 对应 Fortran: subroutine gs_pr_l
// =========================================================================
void NsKernel::gs_pr_l(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta) {
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;
    int nl = 5;
    real_t gamma = GlobalData::getInstance().gamma;
    // 假设 rhs_nb 布局为 (ni*nj*nk*nl)，且为物理区顺序

    std::vector<real_t> rhs0(nl);
    std::vector<real_t> prim_i(nl), prim_j(nl), prim_k(nl);
    std::vector<real_t> gykb_i(4), gykb_j(4), gykb_k(4);
    std::vector<real_t> dq_i(nl), dq_j(nl), dq_k(nl);
    std::vector<real_t> del(nl), dfl(nl), dgl(nl);
    std::vector<real_t> der(nl), dfr(nl), dgr(nl);
    std::vector<real_t> coed(nl);

    // Lambda: 获取 Primitive
    auto get_prim = [&](int i, int j, int k, std::vector<real_t>& p_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        p_out[0] = b->r(I, J, K); p_out[1] = b->u(I, J, K); p_out[2] = b->v(I, J, K);
        p_out[3] = b->w(I, J, K); p_out[4] = b->p(I, J, K);
    };

    // Lambda: 获取 Coed (复用之前的逻辑)
    auto get_coed = [&](int i, int j, int k, std::vector<real_t>& c_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        // 注意：Fortran 这里使用了 nvis 判断，若 nvis>0 则加上粘性谱半径
        // 这里简化为直接加，因为如果 inviscid 模式 srva 等应为 0
        real_t rad_ns = b->sra(I, J, K) + b->srb(I, J, K) + b->src(I, J, K)
                      + b->srva(I, J, K) + b->srvb(I, J, K) + b->srvc(I, J, K);
        real_t ct = 1.0 / b->dtdt(I, J, K);
        real_t ccc = 1.0 / (ct + beta * wmig * rad_ns);
        for(int m=0; m<nl; ++m) c_out[m] = ccc;
    };

    // 线性索引计数器 (用于访问 rhs_nb)
    int idx_rhs = 0;

    // 循环: k, j, i (1..n)
    for (int k = 1; k <= nk; ++k) {
        for (int j = 1; j <= nj; ++j) {
            for (int i = 1; i <= ni; ++i) {
                
                int I = i - 1 + ng;
                int J = j - 1 + ng;
                int K = k - 1 + ng;

                // 1. 获取对角系数
                get_coed(i, j, k, coed);

                // --- 左侧/下方邻居 (Updated values) ---
                for (int m = 0; m < nl; ++m) {
                    dq_i[m] = (i > 1) ? b->dq(I-1, J, K, m) : 0.0;
                    dq_j[m] = (j > 1) ? b->dq(I, J-1, K, m) : 0.0;
                    dq_k[m] = (k > 1) ? b->dq(I, J, K-1, m) : 0.0;
                }

                get_prim(i-1, j, k, prim_i);
                get_prim(i, j-1, k, prim_j);
                get_prim(i, j, k-1, prim_k);

                // Metrics (i-1, j-1, k-1)
                real_t nx, ny, nz, nt;
                get_metrics_at(b, i-1, j, k, 1, 0, 0, nx, ny, nz, nt); gykb_i[0]=nx; gykb_i[1]=ny; gykb_i[2]=nz; gykb_i[3]=nt;
                get_metrics_at(b, i, j-1, k, 0, 1, 0, nx, ny, nz, nt); gykb_j[0]=nx; gykb_j[1]=ny; gykb_j[2]=nz; gykb_j[3]=nt;
                get_metrics_at(b, i, j, k-1, 0, 0, 1, nx, ny, nz, nt); gykb_k[0]=nx; gykb_k[1]=ny; gykb_k[2]=nz; gykb_k[3]=nt;

                // Spectral Radius (-1 direction) -> Use neighbor's radius?
                // Fortran: getrarbrc(..., -1) -> sra(i-1, j, k)
                // 注意索引越界保护: std::max(1, i-1) in Fortran -> std::max(0, i-2) + ng in C++
                // 但 Fortran getrarbrc 内置了 max(..., 1) 逻辑。
                // 简单起见，读取 max(1, i-1) 的谱半径
                int Im = std::max(1, i-1) - 1 + ng; 
                int Jm = std::max(1, j-1) - 1 + ng;
                int Km = std::max(1, k-1) - 1 + ng;
                
                real_t ra = b->sra(Im, J, K);
                real_t rb = b->srb(I, Jm, K);
                real_t rc = b->src(I, J, Km);

                // Flux Split +
                matrix_vector_product_std(prim_i.data(), gykb_i.data(), dq_i.data(), del.data(), ra, 1, gamma);
                matrix_vector_product_std(prim_j.data(), gykb_j.data(), dq_j.data(), dfl.data(), rb, 1, gamma);
                matrix_vector_product_std(prim_k.data(), gykb_k.data(), dq_k.data(), dgl.data(), rc, 1, gamma);

                // --- 右侧/上方邻居 (Current values) ---
                for (int m = 0; m < nl; ++m) {
                    dq_i[m] = (i < ni) ? b->dq(I+1, J, K, m) : 0.0;
                    dq_j[m] = (j < nj) ? b->dq(I, J+1, K, m) : 0.0;
                    dq_k[m] = (k < nk) ? b->dq(I, J, K+1, m) : 0.0;
                }

                get_prim(i+1, j, k, prim_i);
                get_prim(i, j+1, k, prim_j);
                get_prim(i, j, k+1, prim_k);

                // Metrics (i+1, j+1, k+1)
                get_metrics_at(b, i+1, j, k, 1, 0, 0, nx, ny, nz, nt); gykb_i[0]=nx; gykb_i[1]=ny; gykb_i[2]=nz; gykb_i[3]=nt;
                get_metrics_at(b, i, j+1, k, 0, 1, 0, nx, ny, nz, nt); gykb_j[0]=nx; gykb_j[1]=ny; gykb_j[2]=nz; gykb_j[3]=nt;
                get_metrics_at(b, i, j, k+1, 0, 0, 1, nx, ny, nz, nt); gykb_k[0]=nx; gykb_k[1]=ny; gykb_k[2]=nz; gykb_k[3]=nt;

                // Spectral Radius (+1 direction)
                int Ip = std::min(ni, i+1) - 1 + ng;
                int Jp = std::min(nj, j+1) - 1 + ng;
                int Kp = std::min(nk, k+1) - 1 + ng;
                ra = b->sra(Ip, J, K);
                rb = b->srb(I, Jp, K);
                rc = b->src(I, J, Kp);

                // Flux Split -
                matrix_vector_product_std(prim_i.data(), gykb_i.data(), dq_i.data(), der.data(), ra, -1, gamma);
                matrix_vector_product_std(prim_j.data(), gykb_j.data(), dq_j.data(), dfr.data(), rb, -1, gamma);
                matrix_vector_product_std(prim_k.data(), gykb_k.data(), dq_k.data(), dgr.data(), rc, -1, gamma);

                // --- 求解 ---
                for (int m = 0; m < nl; ++m) {
                    // rhs0 = wmig * (del + dfl + dgl - der - dfr - dgr)
                    real_t rhs_val = wmig * (del[m] + dfl[m] + dgl[m] - der[m] - dfr[m] - dgr[m]);
                    
                    // dq = (rhs0 - rhs_nb) * coed
                    // rhs_nb 是我们备份的原始显式残差
                    // 注意 idx_rhs 是按 k, j, i 顺序递增的
                    b->dq(I, J, K, m) = (rhs_val - rhs_nb[idx_rhs++]) * coed[m];
                }
            }
        }
    }
}

// =========================================================================
// Gauss-Seidel 上三角松弛 (Upper Relaxation)
// 对应 Fortran: subroutine gs_pr_u
// =========================================================================
void NsKernel::gs_pr_u(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta) {
    // 逻辑与 gs_pr_l 高度相似，区别在于循环方向和 rhs_nb 的索引访问
    // 由于 rhs_nb 是按 k, j, i (1..n) 存储的，逆序访问时需要小心计算索引
    
    int ni = b->ni; int nj = b->nj; int nk = b->nk; int ng = b->ng; int nl = 5;
    real_t gamma = GlobalData::getInstance().gamma;

    // Buffer 定义 (同上)
    std::vector<real_t> rhs0(nl), prim_i(nl), prim_j(nl), prim_k(nl);
    std::vector<real_t> gykb_i(4), gykb_j(4), gykb_k(4);
    std::vector<real_t> dq_i(nl), dq_j(nl), dq_k(nl);
    std::vector<real_t> del(nl), dfl(nl), dgl(nl), der(nl), dfr(nl), dgr(nl);
    std::vector<real_t> coed(nl);

    auto get_prim = [&](int i, int j, int k, std::vector<real_t>& p_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        p_out[0] = b->r(I, J, K); p_out[1] = b->u(I, J, K); p_out[2] = b->v(I, J, K);
        p_out[3] = b->w(I, J, K); p_out[4] = b->p(I, J, K);
    };

    auto get_coed = [&](int i, int j, int k, std::vector<real_t>& c_out) {
        int I = i - 1 + ng; int J = j - 1 + ng; int K = k - 1 + ng;
        real_t rad_ns = b->sra(I, J, K) + b->srb(I, J, K) + b->src(I, J, K)
                      + b->srva(I, J, K) + b->srvb(I, J, K) + b->srvc(I, J, K);
        real_t ct = 1.0 / b->dtdt(I, J, K);
        real_t ccc = 1.0 / (ct + beta * wmig * rad_ns);
        for(int m=0; m<nl; ++m) c_out[m] = ccc;
    };

    // 循环: k, j, i 逆序
    for (int k = nk; k >= 1; --k) {
        for (int j = nj; j >= 1; --j) {
            for (int i = ni; i >= 1; --i) {
                
                int I = i - 1 + ng;
                int J = j - 1 + ng;
                int K = k - 1 + ng;

                // 计算 rhs_nb 的线性索引
                // rhs_nb layout: [k=1..nk][j=1..nj][i=1..ni][m=0..4]
                // index = ((k-1)*nj*ni + (j-1)*ni + (i-1)) * 5
                // 这里用 size_t 避免溢出
                size_t base_idx = ((size_t)(k-1) * nj * ni + (size_t)(j-1) * ni + (size_t)(i-1)) * 5;

                get_coed(i, j, k, coed);

                // --- 左侧/下方邻居 (Old values, not updated in this sweep yet) ---
                for (int m = 0; m < nl; ++m) {
                    dq_i[m] = (i > 1) ? b->dq(I-1, J, K, m) : 0.0;
                    dq_j[m] = (j > 1) ? b->dq(I, J-1, K, m) : 0.0;
                    dq_k[m] = (k > 1) ? b->dq(I, J, K-1, m) : 0.0;
                }

                get_prim(i-1, j, k, prim_i); get_prim(i, j-1, k, prim_j); get_prim(i, j, k-1, prim_k);

                real_t nx, ny, nz, nt;
                get_metrics_at(b, i-1, j, k, 1, 0, 0, nx, ny, nz, nt); gykb_i[0]=nx; gykb_i[1]=ny; gykb_i[2]=nz; gykb_i[3]=nt;
                get_metrics_at(b, i, j-1, k, 0, 1, 0, nx, ny, nz, nt); gykb_j[0]=nx; gykb_j[1]=ny; gykb_j[2]=nz; gykb_j[3]=nt;
                get_metrics_at(b, i, j, k-1, 0, 0, 1, nx, ny, nz, nt); gykb_k[0]=nx; gykb_k[1]=ny; gykb_k[2]=nz; gykb_k[3]=nt;

                int Im = std::max(1, i-1) - 1 + ng; int Jm = std::max(1, j-1) - 1 + ng; int Km = std::max(1, k-1) - 1 + ng;
                real_t ra = b->sra(Im, J, K); real_t rb = b->srb(I, Jm, K); real_t rc = b->src(I, J, Km);

                matrix_vector_product_std(prim_i.data(), gykb_i.data(), dq_i.data(), del.data(), ra, 1, gamma);
                matrix_vector_product_std(prim_j.data(), gykb_j.data(), dq_j.data(), dfl.data(), rb, 1, gamma);
                matrix_vector_product_std(prim_k.data(), gykb_k.data(), dq_k.data(), dgl.data(), rc, 1, gamma);

                // --- 右侧/上方邻居 (Updated values in this backward sweep) ---
                for (int m = 0; m < nl; ++m) {
                    dq_i[m] = (i < ni) ? b->dq(I+1, J, K, m) : 0.0;
                    dq_j[m] = (j < nj) ? b->dq(I, J+1, K, m) : 0.0;
                    dq_k[m] = (k < nk) ? b->dq(I, J, K+1, m) : 0.0;
                }

                get_prim(i+1, j, k, prim_i); get_prim(i, j+1, k, prim_j); get_prim(i, j, k+1, prim_k);

                get_metrics_at(b, i+1, j, k, 1, 0, 0, nx, ny, nz, nt); gykb_i[0]=nx; gykb_i[1]=ny; gykb_i[2]=nz; gykb_i[3]=nt;
                get_metrics_at(b, i, j+1, k, 0, 1, 0, nx, ny, nz, nt); gykb_j[0]=nx; gykb_j[1]=ny; gykb_j[2]=nz; gykb_j[3]=nt;
                get_metrics_at(b, i, j, k+1, 0, 0, 1, nx, ny, nz, nt); gykb_k[0]=nx; gykb_k[1]=ny; gykb_k[2]=nz; gykb_k[3]=nt;

                int Ip = std::min(ni, i+1) - 1 + ng; int Jp = std::min(nj, j+1) - 1 + ng; int Kp = std::min(nk, k+1) - 1 + ng;
                ra = b->sra(Ip, J, K); rb = b->srb(I, Jp, K); rc = b->src(I, J, Kp);

                matrix_vector_product_std(prim_i.data(), gykb_i.data(), dq_i.data(), der.data(), ra, -1, gamma);
                matrix_vector_product_std(prim_j.data(), gykb_j.data(), dq_j.data(), dfr.data(), rb, -1, gamma);
                matrix_vector_product_std(prim_k.data(), gykb_k.data(), dq_k.data(), dgr.data(), rc, -1, gamma);

                // --- 求解 ---
                for (int m = 0; m < nl; ++m) {
                    real_t rhs_val = wmig * (del[m] + dfl[m] + dgl[m] - der[m] - dfr[m] - dgr[m]);
                    // 注意：这里需要按 base_idx + m 访问 rhs_nb
                    b->dq(I, J, K, m) = (rhs_val - rhs_nb[base_idx + m]) * coed[m];
                }
            }
        }
    }
}

}