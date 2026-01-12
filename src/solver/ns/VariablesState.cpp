#include "NSKernel.h"

namespace Hyres {

void NsKernel::update_derived_variables() {
    // 获取粘性模式配置 (0=Euler, 1=NS)
    int nvis = config_->physics.viscous_mode;

    int rank = mpi_.get_rank();

    // 遍历所有本地块
    for (Block* b : blocks_) {
        if (b && b->owner_rank == rank) {
            // 1. 更新热力学导出量 (声速 c, 温度 T, 粘性 mu)
            if (nvis == 1) {
                // 粘性模式 (NS)
                // 计算声速 c 和 温度 T
                update_thermodynamics(b);
                
                // 计算层流粘性系数 mu_l (依赖 T)
                update_laminar_viscosity(b);
            } else {
                update_sound_speed(b);
            }

            // 3. 更新守恒变量 Q (由 Primitive 计算 Conservative)
            update_conservative_vars(b);

            // 4. 残差/增量数组清零
            reset_residuals(b);
        }
    }
}

void NsKernel::update_thermodynamics(Block* b) {
    const auto& global = GlobalData::getInstance();
    real_t gamma = global.gamma;
    real_t mach  = global.mach;
    real_t mach_sq = mach * mach; // moo2

    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 辅助 Lambda：方便不同层复用逻辑

    // 1. 计算 c 和 t (适用于 Inner 和 n1 层)
    auto calc_c_t = [&](int i, int j, int k) {
        real_t rho = b->r(i, j, k);
        real_t p   = b->p(i, j, k);
        // 假设非 n3 层 rho 肯定为正 (物理区或 n1)
        real_t a2  = gamma * p / rho; 
        
        // 错误检查
        if (a2 <= 0.0) {
            // 仅在 Debug 或 Rank 0 打印，防止刷屏
            // int phys_i = i - ng + 1; ...
            // spdlog::error(...)
            a2 = 1.0e-12;
        }

        b->c(i, j, k) = std::sqrt(a2);
        b->t(i, j, k) = mach_sq * a2;
    };

    // 2. 只计算 t (适用于 n2 层)
    auto calc_t_only = [&](int i, int j, int k) {
        real_t rho = b->r(i, j, k);
        real_t p   = b->p(i, j, k);
        real_t a2  = gamma * p / rho;
        
        if (a2 <= 0.0) a2 = 1.0e-12; // 保护

        b->t(i, j, k) = mach_sq * a2;
    };

    // 3. 只计算 t 且取 abs(rho) (适用于 n3 层)
    auto calc_t_abs = [&](int i, int j, int k) {
        real_t rho = std::abs(b->r(i, j, k)); // 强制取正
        real_t p   = b->p(i, j, k);
        // 加极小值防止 rho=0 (Corner Case)
        real_t a2  = gamma * p / (rho + 1.0e-30);
        
        b->t(i, j, k) = mach_sq * a2;
    };

    // ========================================================
    // 1. 物理核心区域 (Inner Loop)
    // ========================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                calc_c_t(i + ng, j + ng, k + ng);
            }
        }
    }

    // ========================================================
    // 2. I 方向条带 (I-Strips)
    // 范围: J, K 在物理区，I 在 Ghost 区
    // ========================================================
    // Ghost 层的绝对索引
    int n1_L = ng - 1, n1_R = ni + ng;     // Layer 1
    int n2_L = ng - 2, n2_R = ni + ng + 1; // Layer 2
    int n3_L = ng - 3, n3_R = ni + ng + 2; // Layer 3 (最外层)

    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int J = j + ng;
            int K = k + ng;
            
            // n1: 计算 c, t
            calc_c_t(n1_L, J, K);
            calc_c_t(n1_R, J, K);

            // n2: 只计算 t
            calc_t_only(n2_L, J, K);
            calc_t_only(n2_R, J, K);

            // n3: 只计算 t, 取 abs
            calc_t_abs(n3_L, J, K);
            calc_t_abs(n3_R, J, K);
        }
    }

    // ========================================================
    // 3. J 方向条带 (J-Strips)
    // 范围: I, K 在物理区，J 在 Ghost 区
    // ========================================================
    int j_n1_L = ng - 1, j_n1_R = nj + ng;
    int j_n2_L = ng - 2, j_n2_R = nj + ng + 1;
    int j_n3_L = ng - 3, j_n3_R = nj + ng + 2;

    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            int I = i + ng;
            int K = k + ng;

            // n1
            calc_c_t(I, j_n1_L, K);
            calc_c_t(I, j_n1_R, K);
            // n2
            calc_t_only(I, j_n2_L, K);
            calc_t_only(I, j_n2_R, K);
            // n3
            calc_t_abs(I, j_n3_L, K);
            calc_t_abs(I, j_n3_R, K);
        }
    }

    // ========================================================
    // 4. K 方向条带 (K-Strips)
    // 范围: I, J 在物理区，K 在 Ghost 区
    // ========================================================
    int k_n1_L = ng - 1, k_n1_R = nk + ng;
    int k_n2_L = ng - 2, k_n2_R = nk + ng + 1;
    int k_n3_L = ng - 3, k_n3_R = nk + ng + 2;

    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            int I = i + ng;
            int J = j + ng;

            // n1
            calc_c_t(I, J, k_n1_L);
            calc_c_t(I, J, k_n1_R);
            // n2
            calc_t_only(I, J, k_n2_L);
            calc_t_only(I, J, k_n2_R);
            // n3
            calc_t_abs(I, J, k_n3_L);
            calc_t_abs(I, J, k_n3_R);
        }
    }
}

void NsKernel::update_laminar_viscosity(Block* b) {
    // 1. 获取全局单例
    const auto& global = GlobalData::getInstance();
    
    // 2. 获取萨瑟兰常数参数
    real_t visc_const = global.sutherland_c; 

    int ng = b->ng;
    // 范围：全场 (包含物理区和所有 Ghost 层，保证边界通量计算安全)
    int total_i = b->ni + 2 * ng;
    int total_j = b->nj + 2 * ng;
    int total_k = b->nk + 2 * ng;

    for (int k = 0; k < total_k; ++k) {
        for (int j = 0; j < total_j; ++j) {
            for (int i = 0; i < total_i; ++i) {
                
                // 3. 获取温度 tm (已经由 update_thermodynamics 计算好)
                real_t tm = b->t(i, j, k);

                // 4. 安全保护 (防止开根号出现 NaN，或者分母为 0)
                // 这里的 1.0e-30 是为了防止 tm=0 且 visc_const=0 的极端情况
                if (tm < 1.0e-30) tm = 1.0e-30; 

                // 5. 计算粘性 visl (严格复刻公式)
                // Formula: mu = T^1.5 * (1 + S) / (T + S)
                real_t val = tm * std::sqrt(tm) * (1.0 + visc_const) / (tm + visc_const);

                // 6. 存入数组
                b->visl(i, j, k) = val; 
            }
        }
    }
}

void NsKernel::update_sound_speed(Block* b) {
    const auto& global = GlobalData::getInstance();
    real_t gamma = global.gamma;

    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 提取一个 lambda 来复用计算逻辑，避免代码重复
    auto calc_c = [&](int i, int j, int k) {
        real_t rho = b->r(i, j, k);
        real_t p   = b->p(i, j, k);

        // 计算声速平方 a^2 = gamma * p / rho
        // 注意：物理区和第1层Face Ghost的rho肯定为正，无需取abs
        real_t a2 = gamma * p / rho;

        // 错误检查
        if (a2 <= 0.0) {
            // 转换回物理相对索引用于报错
            int phys_i = i - ng + 1;
            int phys_j = j - ng + 1;
            int phys_k = k - ng + 1;
            
            spdlog::error("Negative sound speed sqr at Rank {} Block {}: (I,J,K)=({},{},{}), P={}, Rho={}, a2={}",
                        mpi_.get_rank(), b->id, phys_i, phys_j, phys_k, p, rho, a2);
            
            a2 = 1.0e-12; // 保护
        }

        b->c(i, j, k) = std::sqrt(a2);
    };

    // ========================================================
    // 1. 物理核心区域 (Inner Loop)
    // ========================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                calc_c(i + ng, j + ng, k + ng);
            }
        }
    }

    // ========================================================
    // 2. I 方向条带 (I-Faces, Layer 1 Ghost Only)
    // 对应 Fortran: nn(1)=0, nn(2)=ni+1
    // 范围: I 在边界，J, K 必须在物理区内
    // ========================================================
    int i_L = ng - 1;      // 左边界 (Fortran: 0)
    int i_R = ni + ng;     // 右边界 (Fortran: ni+1)

    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int J = j + ng;
            int K = k + ng;
            calc_c(i_L, J, K);
            calc_c(i_R, J, K);
        }
    }

    // ========================================================
    // 3. J 方向条带 (J-Faces, Layer 1 Ghost Only)
    // 对应 Fortran: nn(1)=0, nn(2)=nj+1
    // 范围: J 在边界，I, K 必须在物理区内
    // ========================================================
    int j_L = ng - 1;
    int j_R = nj + ng;

    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            int I = i + ng;
            int K = k + ng;
            calc_c(I, j_L, K);
            calc_c(I, j_R, K);
        }
    }

    // ========================================================
    // 4. K 方向条带 (K-Faces, Layer 1 Ghost Only)
    // 对应 Fortran: nn(1)=0, nn(2)=nk+1
    // 范围: K 在边界，I, J 必须在物理区内
    // ========================================================
    int k_L = ng - 1;
    int k_R = nk + ng;

    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            int I = i + ng;
            int J = j + ng;
            calc_c(I, J, k_L);
            calc_c(I, J, k_R);
        }
    }
}

void NsKernel::update_conservative_vars(Block* b) {
    const auto& global = GlobalData::getInstance();
    real_t gamma = global.gamma;
    real_t gamma_minus_1_inv = 1.0 / (gamma - 1.0); // 预计算倒数，乘法比除法快

    int ng = b->ng;
    // 遍历全场 (包含物理区和所有 Ghost 层)
    // 这样不仅代码简洁，而且内存访问连续，效率更高
    int total_i = b->ni + 2 * ng;
    int total_j = b->nj + 2 * ng;
    int total_k = b->nk + 2 * ng;

    for (int k = 0; k < total_k; ++k) {
        for (int j = 0; j < total_j; ++j) {
            for (int i = 0; i < total_i; ++i) {
                
                // 1. 读取原始变量
                real_t rho = b->r(i, j, k);
                real_t u   = b->u(i, j, k);
                real_t v   = b->v(i, j, k);
                real_t w   = b->w(i, j, k);
                real_t p   = b->p(i, j, k);

                // 2. 处理负密度 Flag (对应 Fortran prim_to_q_bc)
                // 守恒变量 Q[0] 必须存储带符号的原始密度 (用于传递 Flag)
                // 动量和能量必须使用密度的绝对值计算 (保证物理意义)
                real_t rho_calc = std::abs(rho);

                // 3. 计算动量
                real_t mom_u = rho_calc * u;
                real_t mom_v = rho_calc * v;
                real_t mom_w = rho_calc * w;

                // 4. 计算单位体积总能 rho*E
                // E_total = Internal + Kinetic
                // rho*E = p / (gamma - 1) + 0.5 * rho * V^2
                real_t kinetic_energy = 0.5 * rho_calc * (u * u + v * v + w * w);
                real_t internal_energy = p * gamma_minus_1_inv;
                real_t total_energy_vol = internal_energy + kinetic_energy;

                // 5. 写入守恒变量 Q
                // 假设 Q 的布局是 (i, j, k, var)
                // 0: rho, 1: rhou, 2: rhov, 3: rhow, 4: rhoE
                b->q(i, j, k, 0) = rho;      
                b->q(i, j, k, 1) = mom_u;
                b->q(i, j, k, 2) = mom_v;
                b->q(i, j, k, 3) = mom_w;
                b->q(i, j, k, 4) = total_energy_vol;
            }
        }
    }
}

void NsKernel::reset_residuals(Block* b) {
    int ng = b->ng;
    // 遍历整个内存块 (包含物理区和所有 Ghost 层)
    int total_i = b->ni + 2 * ng;
    int total_j = b->nj + 2 * ng;
    int total_k = b->nk + 2 * ng;
    
    const int n_vars = 5; 

    for (int k = 0; k < total_k; ++k) {
        for (int j = 0; j < total_j; ++j) {
            for (int i = 0; i < total_i; ++i) {
                for (int m = 0; m < n_vars; ++m) {
                    // C++ 0-based 索引，完全覆盖 Fortran 的范围
                    b->dq(i, j, k, m) = 0.0;
                }
            }
        }
    }
}

}
