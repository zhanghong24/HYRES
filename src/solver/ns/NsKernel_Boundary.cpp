#include "NSKernel.h"

namespace Hyres {

// =========================================================================
// 辅助函数：计算单位法向量 (对应 getnxyz_mml)
// =========================================================================
static void get_unit_normal(real_t& nx, real_t& ny, real_t& nz,
                            int i, int j, int k,
                            int is_step, int js_step, int ks_step,
                            const Block* b) {
    // 1. 初始化
    nx = 0.0; ny = 0.0; nz = 0.0;

    // 2. 累加 Xi 方向的度量项 (n=1 to abs(is))
    // 注意：is_step, js_step, ks_step 实际上只有其中一个是 1 (或 -1)，其余为 0
    // 所以这里的循环本质上就是选出哪个方向是法向
    
    // 如果法向是 I (Xi)
    if (is_step != 0) {
        nx += b->kcx(i, j, k);
        ny += b->kcy(i, j, k);
        nz += b->kcz(i, j, k);
    }

    // 如果法向是 J (Eta)
    if (js_step != 0) {
        nx += b->etx(i, j, k);
        ny += b->ety(i, j, k);
        nz += b->etz(i, j, k);
    }

    // 如果法向是 K (Zeta)
    if (ks_step != 0) {
        nx += b->ctx(i, j, k);
        ny += b->cty(i, j, k);
        nz += b->ctz(i, j, k);
    }

    // 3. 归一化
    // 这里的 sml_sss 是防止除零的小量，通常定义在 GlobalData 中，这里硬编码或用宏
    const real_t small_eps = 1.0e-38; 
    real_t mag = std::sqrt(nx*nx + ny*ny + nz*nz);
    mag = std::max(mag, small_eps);

    nx /= mag;
    ny /= mag;
    nz /= mag;
}

// =========================================================================
// 序列处理：按优先级顺序执行边界
// =========================================================================
void NsKernel::BoundaryManager::boundary_sequence(Block* b) {
    for (int nr : b->bc_execution_order) {
        BoundaryPatch& patch = b->boundaries[nr];
        if (patch.type < 0) {
            boundary_n1_vir3_parallel(b, patch);
        }
        else if (patch.type == 2) {
            if (config_->physics.viscous_mode == 0) {
                boundary_inviscid_wall(b, patch);
            } else {
                boundary_viscid_wall(b, patch);
            }
        }
        else if (patch.type == 3) {
            boundary_symmetry(b, patch);
        }
        else if (patch.type == 4) {
            boundary_farfield(b, patch);
        }
        else if (patch.type == 5) {
            boundary_freestream(b, patch);
        }
        else if (patch.type == 6) {
            boundary_outflow(b, patch);
        }
    }
}

// =========================================================================
// 辅助函数：差分格式的边界平均修正
// =========================================================================
void NsKernel::BoundaryManager::dif_average(Block* b, const BoundaryPatch& patch) {
    int idir = patch.s_nd; // 0=I, 1=J, 2=K
    int inrout = patch.s_lr; // -1 or 1

    int ng = b->ng;
    
    // 获取循环范围 (0-based)
    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);
    
    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;

    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 1. 边界点本身 (i,j,k)
                int i0 = i + ng - 1; // 物理索引 (注意：这里需要微调)
                int j0 = j + ng - 1;
                int k0 = k + ng - 1;

                // 修正：上述循环范围 ist..ied 是 1-based 的物理索引
                // 转为 C++ 数组索引：
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // 2. Ghost Cell (Layer 1)
                int is = idx_i + step_s[0];
                int js = idx_j + step_s[1];
                int ks = idx_k + step_s[2];

                // 3. Internal Cell (Layer 1, inside)
                int it = idx_i - step_s[0];
                int jt = idx_j - step_s[1];
                int kt = idx_k - step_s[2];

                // 4. 执行平均
                b->r(idx_i, idx_j, idx_k) = 0.5 * (b->r(is, js, ks) + b->r(it, jt, kt));
                b->u(idx_i, idx_j, idx_k) = 0.5 * (b->u(is, js, ks) + b->u(it, jt, kt));
                b->v(idx_i, idx_j, idx_k) = 0.5 * (b->v(is, js, ks) + b->v(it, jt, kt));
                b->w(idx_i, idx_j, idx_k) = 0.5 * (b->w(is, js, ks) + b->w(it, jt, kt));
                b->p(idx_i, idx_j, idx_k) = 0.5 * (b->p(is, js, ks) + b->p(it, jt, kt));
            }
        }
    }
}

// =========================================================================
// 物理边界：无粘壁面 (Slip Wall / Euler Wall)
// =========================================================================
void NsKernel::BoundaryManager::boundary_inviscid_wall(Block* b, const BoundaryPatch& patch) {
    int idir = patch.s_nd; // 0, 1, 2
    int inrout = patch.s_lr; // -1, 1

    int ng = b->ng;
    
    // 循环范围 (1-based from file -> 0-based loops)
    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);

    // 法向量步进方向 (用于 get_unit_normal)
    // 对应 Fortran: s_lr3d
    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;

    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 当前物理边界点索引 (0-based + ghost offset)
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // 1. 计算该点的单位法向量 (nx, ny, nz)
                // 注意：Fortran 传的是 t_nd, t_fix, t_lr，实际上就是法向向量分量
                // 这里我们直接传 step_s 即可
                real_t nx, ny, nz;
                get_unit_normal(nx, ny, nz, idx_i, idx_j, idx_k, 
                                step_s[0], step_s[1], step_s[2], b);

                // 2. 循环填充 Ghost Layers
                for (int l = 1; l <= ng; ++l) {
                    // Ghost Index (向外)
                    int is = idx_i + l * step_s[0];
                    int js = idx_j + l * step_s[1];
                    int ks = idx_k + l * step_s[2];

                    // Internal Image Index (向内镜像)
                    int it = idx_i - l * step_s[0];
                    int jt = idx_j - l * step_s[1];
                    int kt = idx_k - l * step_s[2];

                    // 获取内部点的速度
                    real_t u_in = b->u(it, jt, kt);
                    real_t v_in = b->v(it, jt, kt);
                    real_t w_in = b->w(it, jt, kt);

                    // 计算法向速度分量 (dot product)
                    // sss1 = 2 * (V . n)
                    real_t v_dot_n = u_in * nx + v_in * ny + w_in * nz;
                    real_t sss1 = 2.0 * v_dot_n;

                    // 填充标量 (p, r)
                    b->p(is, js, ks) = b->p(it, jt, kt);

                    // 关键逻辑：第3层 Ghost 密度取负值 (Flag for shock capturing)
                    // 注意：这取决于你的 ng 是多少。通常只在最后一层做这个标记。
                    if (l == ng) {
                         b->r(is, js, ks) = -1.0 * b->r(it, jt, kt);
                    } else {
                         b->r(is, js, ks) =  b->r(it, jt, kt);
                    }

                    // 填充矢量 (Reflect Velocity)
                    // V_ghost = V_in - 2*(V_in . n) * n
                    b->u(is, js, ks) = u_in - sss1 * nx;
                    b->v(is, js, ks) = v_in - sss1 * ny;
                    b->w(is, js, ks) = w_in - sss1 * nz;
                }
            }
        }
    }

    // 3. 差分修正
    dif_average(b, patch);
}

// =========================================================================
// 物理边界：粘性壁面 (No-Slip Wall) 
// 支持：绝热 (Adiabatic) / 等温 (Isothermal) + 鲁棒性限幅
// =========================================================================
void NsKernel::BoundaryManager::boundary_viscid_wall(Block* b, const BoundaryPatch& patch) {
    // 1. 获取全局单例数据和配置
    const auto& global = GlobalData::getInstance();
    real_t mach  = global.mach;
    real_t gamma = global.gamma;
    real_t t_ref = global.t_ref;
    real_t t_wall_input = config_->inflow.T_wall; 

    // 物理限幅
    const real_t pmin_limit = 1.0e-8;
    const real_t tmin_limit = 1.0e-8;
    
    real_t moocp = mach * mach * gamma;

    // 2. 几何方向准备
    int idir = patch.s_nd;
    int inrout = patch.s_lr;
    int ng = b->ng;

    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);

    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;

    // 3. 判断壁面热力学状态
    bool is_isothermal = (t_wall_input > 0.0);
    real_t t_wall_nondim = 0.0;
    if (is_isothermal) {
        real_t safe_tref = (std::abs(t_ref) < 1.0e-10) ? 288.15 : t_ref;
        t_wall_nondim = t_wall_input / safe_tref;
    }

    // 4. 执行边界遍历
    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 当前物理边界点索引 (Face Node, Layer 0)
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // [关键修正] 锁定最靠近壁面的第一层内部点 (Inner Layer 1)
                // Fortran 逻辑: 所有的 Ghost 层都参考这一个点
                int it_i = idx_i - step_s[0];
                int it_j = idx_j - step_s[1];
                int it_k = idx_k - step_s[2];

                // ========================================================
                // Part A: 填充 Ghost Cells (ng 层) - 复刻 Fortran 逻辑
                // ========================================================
                for (int l = 1; l <= ng; ++l) {
                    // Ghost Index (向外延伸)
                    int is_i = idx_i + l * step_s[0];
                    int is_j = idx_j + l * step_s[1];
                    int is_k = idx_k + l * step_s[2];

                    // 注意：这里不再更新 it，始终使用 Inner Layer 1 的值
                    
                    // 1. 标量对称
                    b->r(is_i, is_j, is_k) = b->r(it_i, it_j, it_k);
                    b->p(is_i, is_j, is_k) = b->p(it_i, it_j, it_k);
                    b->t(is_i, is_j, is_k) = b->t(it_i, it_j, it_k);

                    // 2. 速度反对称 (No-Slip)
                    b->u(is_i, is_j, is_k) = -b->u(it_i, it_j, it_k);
                    b->v(is_i, is_j, is_k) = -b->v(it_i, it_j, it_k);
                    b->w(is_i, is_j, is_k) = -b->w(it_i, it_j, it_k);

                    // 3. 【Flag】最外层 Ghost 密度取负 (Fortran 逻辑)
                    // Fortran: 最后一层 Ghost 的密度取负值 (仍然基于 Inner Layer 1)
                    if (l == ng) {
                        b->r(is_i, is_j, is_k) = -b->r(it_i, it_j, it_k);
                    }
                }

                // ========================================================
                // Part B: 修正物理边界点本身 (Face Node / Layer 0)
                // ========================================================
                // 这部分逻辑 Fortran 和 C++ 原版基本一致，不需要大改
                
                int it2_i = idx_i - 2 * step_s[0]; 
                int it2_j = idx_j - 2 * step_s[1]; 
                int it2_k = idx_k - 2 * step_s[2];

                real_t p1 = b->p(it_i, it_j, it_k);
                real_t p2 = b->p(it2_i, it2_j, it2_k);
                real_t pres = (4.0 * p1 - p2) / 3.0;
                if (pres < pmin_limit) pres = p1;

                real_t tres = 0.0;
                if (is_isothermal) {
                    tres = t_wall_nondim;
                } else {
                    real_t t1 = b->t(it_i, it_j, it_k);
                    real_t t2 = b->t(it2_i, it2_j, it2_k);
                    tres = (4.0 * t1 - t2) / 3.0;
                    if (tres < tmin_limit) tres = t1;
                }

                real_t dres = moocp * pres / tres;

                // 赋值回 Face
                b->r(idx_i, idx_j, idx_k) = dres;
                b->u(idx_i, idx_j, idx_k) = 0.0;
                b->v(idx_i, idx_j, idx_k) = 0.0;
                b->w(idx_i, idx_j, idx_k) = 0.0;
                b->p(idx_i, idx_j, idx_k) = pres;
                b->t(idx_i, idx_j, idx_k) = tres; 
            }
        }
    }
}

// =========================================================================
// 物理边界：对称边界 (Symmetry)
// 逻辑：计算边界面中心点的法向量，应用于全场进行镜像反射
// =========================================================================
void NsKernel::BoundaryManager::boundary_symmetry(Block* b, const BoundaryPatch& patch) {
    // 1. 几何方向准备
    int idir = patch.s_nd;   // 0=I, 1=J, 2=K
    int inrout = patch.s_lr; // -1, 1
    int ng = b->ng;

    // 循环范围 (1-based from file -> 0-based loops)
    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);

    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;

    // ========================================================
    // Part A: 计算边界面的“平均”法向量 (取几何中心点)
    // ========================================================
    real_t nx = 0.0, ny = 0.0, nz = 0.0;
    
    // 计算中心点索引 (Fortran: (st+ed)/2)
    int i_mid_raw = (ist + ied) / 2;
    int j_mid_raw = (jst + jed) / 2;
    int k_mid_raw = (kst + ked) / 2;

    // 转换为 Block 内部索引 (0-based + ng)
    int idx_i_mid = (i_mid_raw - 1) + ng;
    int idx_j_mid = (j_mid_raw - 1) + ng;
    int idx_k_mid = (k_mid_raw - 1) + ng;

    // 调用之前实现的 get_unit_normal (复用代码)
    // 注意：这里传入 step_s，函数内部会根据 step_s 决定累加哪个方向的 Metrics
    get_unit_normal(nx, ny, nz, idx_i_mid, idx_j_mid, idx_k_mid, 
                    step_s[0], step_s[1], step_s[2], b);

    // ========================================================
    // Part B: 遍历边界，使用统一的法向量进行填充
    // ========================================================
    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 当前物理边界点索引
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // 循环填充所有 Ghost 层
                for (int l = 1; l <= ng; ++l) {
                    // Ghost Index (向外)
                    int is = idx_i + l * step_s[0];
                    int js = idx_j + l * step_s[1];
                    int ks = idx_k + l * step_s[2];

                    // Internal Image Index (向内镜像)
                    int it = idx_i - l * step_s[0];
                    int jt = idx_j - l * step_s[1];
                    int kt = idx_k - l * step_s[2];

                    // 1. 获取内部数据
                    real_t u_in = b->u(it, jt, kt);
                    real_t v_in = b->v(it, jt, kt);
                    real_t w_in = b->w(it, jt, kt);

                    // 2. 计算反射量 (Householder Reflection)
                    // 使用 Part A 计算出的统一法向量 (nx, ny, nz)
                    // sss1 = 2 * (V . n)
                    real_t dot_prod = u_in * nx + v_in * ny + w_in * nz;
                    real_t sss1 = 2.0 * dot_prod;

                    // 3. 填充标量 (对称)
                    // 注意：Symmetry 边界通常不反转密度符号
                    b->r(is, js, ks) = b->r(it, jt, kt);
                    b->p(is, js, ks) = b->p(it, jt, kt);
                    b->t(is, js, ks) = b->t(it, jt, kt);

                    // 4. 填充速度 (反射)
                    b->u(is, js, ks) = u_in - sss1 * nx;
                    b->v(is, js, ks) = v_in - sss1 * ny;
                    b->w(is, js, ks) = w_in - sss1 * nz;
                }
            }
        }
    }

    // ========================================================
    // Part C: 差分修正 (method == 1)
    // ========================================================
    // 修正物理边界点本身 (Layer 0)
    dif_average(b, patch);
}

// =========================================================================
// 物理边界：远场 (Farfield) - 基于黎曼不变量 (Riemann Invariants)
// =========================================================================
void NsKernel::BoundaryManager::boundary_farfield(Block* b, const BoundaryPatch& patch) {
    const auto& global = GlobalData::getInstance();
    
    // --- 1. 获取自由来流参数 (Free Stream / Infinity) ---
    real_t r_inf = global.rho_inf;      // roo
    real_t u_inf = global.u_inf;        // uoo
    real_t v_inf = global.v_inf;        // voo
    real_t w_inf = global.w_inf;        // woo
    real_t p_inf = global.p_inf;        // poo
    real_t gamma = global.gamma;        // gama
    
    // 预计算远场声速 (c_inf) 和 熵 (s_inf = P / rho^gamma)
    real_t c_inf = std::sqrt(gamma * p_inf / r_inf);
    real_t s_inf = p_inf / std::pow(r_inf, gamma);

    // --- 2. 几何与循环准备 ---
    int idir = patch.s_nd;   // 0=I, 1=J, 2=K
    int inrout = patch.s_lr; // -1, 1
    int ng = b->ng;

    // 循环范围 (Plot3D 1-based -> 0-based loops)
    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);

    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;
    
    // sml_sss 用于防止除零
    const real_t sml_sss = 1.0e-30;

    // --- 3. 开始遍历 ---
    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 当前物理边界点 (Face Node, Layer 0)
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // 紧邻的内部点 (Inner Node)
                int it = idx_i - step_s[0];
                int jt = idx_j - step_s[1];
                int kt = idx_k - step_s[2];

                // --- 4. 计算单位法向量 (指向外) ---
                real_t nx, ny, nz;
                get_unit_normal(nx, ny, nz, idx_i, idx_j, idx_k, 
                                step_s[0], step_s[1], step_s[2], b);
                
                // 注意：Fortran 代码中做了 nxa = nx * (lr / |n|) 的操作
                // 实际上 get_unit_normal 返回的是几何梯度方向
                // 我们需要确保法向量是指向计算域“外”的
                // step_s[idir] (即 inrout) 为 1 代表右/上/后边界，法向本来就对外
                // step_s[idir] 为 -1 代表左/下/前边界，法向需要反向才对外
                // 这里的逻辑：我们统一让 (nx,ny,nz) 指向外部
                if (inrout < 0) {
                    nx = -nx; ny = -ny; nz = -nz;
                }

                // --- 5. 获取内部状态 ---
                real_t r_in = b->r(it, jt, kt);
                real_t u_in = b->u(it, jt, kt);
                real_t v_in = b->v(it, jt, kt);
                real_t w_in = b->w(it, jt, kt);
                real_t p_in = b->p(it, jt, kt);

                real_t c_in = std::sqrt(gamma * p_in / r_in);
                real_t s_in = p_in / std::pow(r_in, gamma);

                // --- 6. 计算法向速度 ---
                real_t vn_inf = nx * u_inf + ny * v_inf + nz * w_inf;
                real_t vn_in  = nx * u_in  + ny * v_in  + nz * w_in;

                // --- 7. 判断流态 (Mach Number) ---
                real_t vel_mag_sq = u_in*u_in + v_in*v_in + w_in*w_in;
                real_t mach_local = std::sqrt(vel_mag_sq) / c_in;

                // 边界值变量
                real_t rb, ub, vb, wb, pb;

                if (mach_local < 1.0) {
                    // === 亚声速：黎曼不变量 (Riemann Invariants) ===
                    
                    // R+ (outgoing) 和 R- (incoming)
                    // 注意：这里的公式假设法向指向外。
                    // 黎曼不变量通常定义为 V_n +/- 2c/(g-1)
                    real_t r_plus  = vn_in  + 2.0 * c_in  / (gamma - 1.0);
                    real_t r_minus = vn_inf - 2.0 * c_inf / (gamma - 1.0);

                    // 边界法向速度和声速
                    real_t vn_b = 0.5 * (r_plus + r_minus);
                    real_t c_b  = 0.25 * (gamma - 1.0) * (r_plus - r_minus);

                    // 熵和切向速度的选择 (根据流向)
                    real_t s_b, u_ref, v_ref, w_ref, vn_ref;
                    
                    if (vn_b > 0.0) { // 流出 (Outflow) -> 取内部信息
                        s_b    = s_in;
                        u_ref  = u_in; v_ref = v_in; w_ref = w_in;
                        vn_ref = vn_in;
                    } else {          // 流入 (Inflow) -> 取外部信息
                        s_b    = s_inf;
                        u_ref  = u_inf; v_ref = v_inf; w_ref = w_inf;
                        vn_ref = vn_inf;
                    }

                    // 反算边界状态
                    // rho = ( c^2 / (gamma * s) ) ^ (1/(g-1))
                    // Fortran: rb = ( ab*ab/gama/sb ) ** ( 1.0/(gama-1.0) )  where ab ~ c
                    // 这里 c_b 就是 ab (声速)
                    rb = std::pow(c_b * c_b / (gamma * s_b), 1.0 / (gamma - 1.0));
                    pb = s_b * std::pow(rb, gamma);

                    // 速度矢量重构: V_b = V_ref + (Vn_b - Vn_ref) * n
                    ub = u_ref + nx * (vn_b - vn_ref);
                    vb = v_ref + ny * (vn_b - vn_ref);
                    wb = w_ref + nz * (vn_b - vn_ref);

                } else {
                    // === 超声速：简单迎风 ===
                    if (vn_in > 0.0) {
                        // 超声速流出 (Outflow) -> 全部外推内部
                        rb = r_in; ub = u_in; vb = v_in; wb = w_in; pb = p_in;
                    } else {
                        // 超声速流入 (Inflow) -> 全部强制为来流
                        rb = r_inf; ub = u_inf; vb = v_inf; wb = w_inf; pb = p_inf;
                    }
                }

                // --- 8. 填充 Ghost Cells ---
                // 注意：Fortran 代码中，boundary_4 把 Face Node (is, js, ks) 也填充了
                // 并且 is1 = i + s_lr * n (n=1..2)，也就是填充了 Face, Ghost1, Ghost2
                
                // 我们统一逻辑：先填 Face，再填 Ghost
                
                // 8.1 填充 Face Node (Layer 0)
                // 这一步类似于 dif_average，但使用的是 Riemann 解而非简单平均
                b->r(idx_i, idx_j, idx_k) = rb;
                b->u(idx_i, idx_j, idx_k) = ub;
                b->v(idx_i, idx_j, idx_k) = vb;
                b->w(idx_i, idx_j, idx_k) = wb;
                b->p(idx_i, idx_j, idx_k) = pb;

                // 8.2 填充 Ghost Layers (Layer 1 to ng)
                for (int l = 1; l <= ng; ++l) {
                    int is = idx_i + l * step_s[0];
                    int js = idx_j + l * step_s[1];
                    int ks = idx_k + l * step_s[2];

                    // 最后一层 Ghost 密度取负 (Flag)
                    if (l == ng) {
                        b->r(is, js, ks) = -rb;
                    } else {
                        b->r(is, js, ks) = rb;
                    }
                    
                    b->u(is, js, ks) = ub;
                    b->v(is, js, ks) = vb;
                    b->w(is, js, ks) = wb;
                    b->p(is, js, ks) = pb;
                }
            } // i
        } // j
    } // k
}

// =========================================================================
// 物理边界：自由流/强迫入口 (Freestream / Inflow)
// 逻辑：强制设为无穷远来流参数
// =========================================================================
void NsKernel::BoundaryManager::boundary_freestream(Block* b, const BoundaryPatch& patch) {
    const auto& global = GlobalData::getInstance();

    // 1. 获取全局来流参数
    real_t r_inf = global.rho_inf;
    real_t u_inf = global.u_inf;
    real_t v_inf = global.v_inf;
    real_t w_inf = global.w_inf;
    real_t p_inf = global.p_inf;
    // 温度和能量通常由上述变量导出，但在 Plot3D 原始变量存储中主要存 r,u,v,w,p

    // 2. 几何方向准备
    int idir = patch.s_nd;   // 0=I, 1=J, 2=K
    int inrout = patch.s_lr; // -1, 1
    int ng = b->ng;

    // 循环范围 (1-based -> 0-based)
    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);

    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;

    // 3. 遍历边界
    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 当前物理边界点索引 (Face Node)
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // 4.1 填充 Face Node (Layer 0)
                b->r(idx_i, idx_j, idx_k) = r_inf;
                b->u(idx_i, idx_j, idx_k) = u_inf;
                b->v(idx_i, idx_j, idx_k) = v_inf;
                b->w(idx_i, idx_j, idx_k) = w_inf;
                b->p(idx_i, idx_j, idx_k) = p_inf;

                // 4.2 填充 Ghost Layers (Layer 1 to ng)
                for (int l = 1; l <= ng; ++l) {
                    int is = idx_i + l * step_s[0];
                    int js = idx_j + l * step_s[1];
                    int ks = idx_k + l * step_s[2];

                    // 注意：Fortran boundary_5 中所有 Ghost 密度都是正值 (roo)
                    // 没有取负操作
                    b->r(is, js, ks) = r_inf;
                    b->u(is, js, ks) = u_inf;
                    b->v(is, js, ks) = v_inf;
                    b->w(is, js, ks) = w_inf;
                    b->p(is, js, ks) = p_inf;
                }
            }
        }
    }
}

// =========================================================================
// 物理边界：超声速出口 / 零阶外推 (Supersonic Outflow / Zero-Order Extrapolation)
// 逻辑：Ghost = Face = Inner (直接复制最近的内部点)
// =========================================================================
void NsKernel::BoundaryManager::boundary_outflow(Block* b, const BoundaryPatch& patch) {
    // 1. 几何方向准备
    int idir = patch.s_nd;   
    int inrout = patch.s_lr; 
    int ng = b->ng;

    // 循环范围
    int ist = std::abs(patch.raw_is[0]); int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]); int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]); int ked = std::abs(patch.raw_ie[2]);

    int step_s[3] = {0, 0, 0};
    step_s[idir] = inrout;

    // 2. 遍历边界
    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // Face Node 索引
                int idx_i = (i - 1) + ng;
                int idx_j = (j - 1) + ng;
                int idx_k = (k - 1) + ng;

                // 获取最近的内部点 (First Inner Point)
                // 对应 Fortran: it = i - s_lr3d * method (假设 method=1, 则取 i-1)
                // 在 C++ 0-based 中，这就是 idx_i - step
                int it_i = idx_i - step_s[0];
                int it_j = idx_j - step_s[1];
                int it_k = idx_k - step_s[2];

                // 读取内部值 (Zero-order extrapolation)
                real_t r_in = b->r(it_i, it_j, it_k);
                real_t u_in = b->u(it_i, it_j, it_k);
                real_t v_in = b->v(it_i, it_j, it_k);
                real_t w_in = b->w(it_i, it_j, it_k);
                real_t p_in = b->p(it_i, it_j, it_k);

                // 3.1 填充 Face Node (Layer 0)
                b->r(idx_i, idx_j, idx_k) = r_in;
                b->u(idx_i, idx_j, idx_k) = u_in;
                b->v(idx_i, idx_j, idx_k) = v_in;
                b->w(idx_i, idx_j, idx_k) = w_in;
                b->p(idx_i, idx_j, idx_k) = p_in;

                // 3.2 填充 Ghost Layers (Layer 1 to ng)
                for (int l = 1; l <= ng; ++l) {
                    int is = idx_i + l * step_s[0];
                    int js = idx_j + l * step_s[1];
                    int ks = idx_k + l * step_s[2];

                    // 注意：Fortran boundary_6 的最后一段注释写了 "! 3ܶΪ" (正值?)
                    // 实际上代码里写的是 `mb_r... = rm` (正值)
                    // 这意味着 Outflow 边界通常不需要取负密度标记
                    b->r(is, js, ks) = r_in; 
                    b->u(is, js, ks) = u_in;
                    b->v(is, js, ks) = v_in;
                    b->w(is, js, ks) = w_in;
                    b->p(is, js, ks) = p_in;
                }
            }
        }
    }
}

void NsKernel::BoundaryManager::boundary_n1_vir3_parallel(Block* b, BoundaryPatch& patch) {
    int nbt_id = patch.target_block;
    int id_src = b->owner_rank;
    int id_des = blocks_[nbt_id]->owner_rank;

    if (id_src == id_des) {
        boundary_n1_vir3(b, patch);
    } else {
        boundary_n1_vir3_other(b, patch);
    }
}

// =========================================================================
// 核心逻辑 1：本地拷贝
// =========================================================================
void NsKernel::BoundaryManager::boundary_n1_vir3(Block* b, BoundaryPatch& patch) {
    int nbt_id = patch.target_block;
    Block* b_target = blocks_[nbt_id];

    // 获取法向信息
    int idir = patch.s_nd; // 0, 1, 2
    int inrout = patch.s_lr; // -1, 1

    // 目标 Patch 索引
    int target_patch_idx = patch.window_id - 1;
    BoundaryPatch& t_patch = b_target->boundaries[target_patch_idx];
    int t_idir = t_patch.s_nd;
    int t_inrout = t_patch.s_lr;

    int ng = b->ng;
    int ng_t = b_target->ng;
    real_t cic1 = 0.0; // 对应 Fortran cic1

    // 遍历边界面，利用 raw_is/ie 的绝对值确定循环范围
    int ist = std::abs(patch.raw_is[0]);
    int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]);
    int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]);
    int ked = std::abs(patch.raw_ie[2]);

    auto patch_size = patch.get_size();

    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                // 1. 映射数组索引 (0-based)
                int bi = i - ist;
                int bj = j - jst;
                int bk = k - kst;
                int flat_idx = bi + patch_size[0] * (bj + patch_size[1] * bk);

                // 2. 获取目标点的物理 0-based 索引
                int it0_0 = patch.image[flat_idx];
                int jt0_0 = patch.jmage[flat_idx];
                int kt0_0 = patch.kmage[flat_idx];

                // 3. 计算本地方向向量步进 (s_lr3d 逻辑)
                int step_s[3] = {0, 0, 0};
                step_s[idir] = inrout;

                // 4. 计算目标方向向量步进 (t_lr3d 逻辑)
                int step_t[3] = {0, 0, 0};
                step_t[t_idir] = t_inrout;

                // 5. 循环 ng 层填充
                for (int layer = 1; layer <= ng; ++layer) {
                    // 源块 Ghost 坐标 (0-based index + ng)
                    int is = (i - 1) + ng + layer * step_s[0];
                    int js = (j - 1) + ng + layer * step_s[1];
                    int ks = (k - 1) + ng + layer * step_s[2];

                    // 目标块内部坐标
                    int it = it0_0 + ng_t - layer * step_t[0];
                    int jt = jt0_0 + ng_t - layer * step_t[1];
                    int kt = kt0_0 + ng_t - layer * step_t[2];

                    real_t factor = (layer == ng) ? (1.0 - 2.0 * cic1) : 1.0;

                    b->r(is, js, ks) = b_target->r(it, jt, kt) * factor;
                    b->u(is, js, ks) = b_target->u(it, jt, kt);
                    b->v(is, js, ks) = b_target->v(it, jt, kt);
                    b->w(is, js, ks) = b_target->w(it, jt, kt);
                    b->p(is, js, ks) = b_target->p(it, jt, kt);
                }
            }
        }
    }
}

// =========================================================================
// 核心逻辑 2：远程解包
// =========================================================================
void NsKernel::BoundaryManager::boundary_n1_vir3_other(Block* b, BoundaryPatch& patch) {
    int idir = patch.s_nd;
    int inrout = patch.s_lr;

    int nbt_id = patch.target_block;
    int target_patch_idx = patch.window_id - 1;
    BoundaryPatch& t_patch = blocks_[nbt_id]->boundaries[target_patch_idx];
    int t_idir = t_patch.s_nd;
    int t_inrout = t_patch.s_lr;

    // 计算发送方的 ibeg (完全匹配 exchange_bc 打包逻辑)
    int r_ibeg[3];
    {
        const int mcyc[5] = {0, 1, 2, 0, 1};
        int tmp_ibeg[3];
        for(int k=0; k<3; ++k) tmp_ibeg[k] = std::abs(t_patch.raw_is[k]);
        int t_idir_pack = t_patch.s_nd;
        int t_inrout_pack = t_patch.s_lr;
        tmp_ibeg[t_idir_pack] -= 4 * std::max(t_inrout_pack, 0); 
        for (int m = 1; m <= 2; ++m) {
            int idir2 = mcyc[t_idir_pack + m];
            if (tmp_ibeg[idir2] > 1) tmp_ibeg[idir2] -= 1;
        }
        for(int k=0; k<3; ++k) r_ibeg[k] = tmp_ibeg[k];
    }

    int ng = b->ng;
    auto& qpvpack = patch.qpvpack;

    int ist = std::abs(patch.raw_is[0]);
    int ied = std::abs(patch.raw_ie[0]);
    int jst = std::abs(patch.raw_is[1]);
    int jed = std::abs(patch.raw_ie[1]);
    int kst = std::abs(patch.raw_is[2]);
    int ked = std::abs(patch.raw_ie[2]);

    auto patch_size = patch.get_size();

    for (int k = kst; k <= ked; ++k) {
        for (int j = jst; j <= jed; ++j) {
            for (int i = ist; i <= ied; ++i) {
                
                int bi = i - ist;
                int bj = j - jst;
                int bk = k - kst;
                int flat_idx = bi + patch_size[0] * (bj + patch_size[1] * bk);
                
                int it0_0 = patch.image[flat_idx];
                int jt0_0 = patch.jmage[flat_idx];
                int kt0_0 = patch.kmage[flat_idx];

                int step_s[3] = {0, 0, 0};
                step_s[idir] = inrout;

                int step_t[3] = {0, 0, 0};
                step_t[t_idir] = t_inrout;

                for (int layer = 1; layer <= ng; ++layer) {
                    int is = (i - 1) + ng + layer * step_s[0];
                    int js = (j - 1) + ng + layer * step_s[1];
                    int ks = (k - 1) + ng + layer * step_s[2];

                    // Buffer 中的逻辑坐标 (1-based)
                    int it_l = (it0_0 + 1) - layer * step_t[0];
                    int jt_l = (jt0_0 + 1) - layer * step_t[1];
                    int kt_l = (kt0_0 + 1) - layer * step_t[2];

                    // 计算 Buffer 索引
                    int qi = it_l - r_ibeg[0];
                    int qj = jt_l - r_ibeg[1];
                    int qk = kt_l - r_ibeg[2];

                    b->r(is, js, ks) = qpvpack(qi, qj, qk, 0);
                    b->u(is, js, ks) = qpvpack(qi, qj, qk, 1);
                    b->v(is, js, ks) = qpvpack(qi, qj, qk, 2);
                    b->w(is, js, ks) = qpvpack(qi, qj, qk, 3);
                    b->p(is, js, ks) = qpvpack(qi, qj, qk, 4);
                }
            }
        }
    }
}

}