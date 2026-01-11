#include "solver/SolverDriver.h"
#include "spdlog/spdlog.h"
#include "data/GlobalData.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <limits>

namespace Hyres {

SolverDriver::SolverDriver(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi)
    : blocks_(blocks), config_(config), mpi_(mpi), boundary_manager_(config, blocks) {
    
    current_step_ = config->control.start_mode == 0 ? 0 : 0; // TODO: Handle restart step
    max_steps_ = config->control.max_steps;
}

SolverDriver::~SolverDriver() = default;

void SolverDriver::solve() {
    if (mpi_.is_root()) {
        spdlog::info(">>> Solver Started. Max Steps: {}", max_steps_);
    }

    // 主时间步循环 (Main Loop)
    for (int step = current_step_ + 1; step <= max_steps_; ++step) {
        
        // 1. 执行单步时间推进 (RK3, LU-SGS, etc.)
        run_time_step(step);

        // 2. 监控残差 (Residual Monitor)
        if (step % config_->control.res_interval == 0) {
            check_residual(step);
        }

        // 3. 升阻力输出
        if (step % config_->control.force_interval == 0) {
            calculate_aerodynamic_forces(step);
        }

        // 3. I/O 输出 (Result Saving)
        if (step % config_->control.save_interval == 0) {
            output_solution(step);
        }
    }

    if (mpi_.is_root()) {
        spdlog::info(">>> Solver Finished Successfully.");
    }
}

void SolverDriver::run_time_step(int step) {
    // 1. 交换边界信息(rho, u, v, w, p)
    exchange_bc();

    // 2. 填充 Ghost Cell (对应 boundary_all)
    boundary_all();

    // 3. 更新守恒量与声速
    update_derived_variables();

    // 4. 计算时间步长
    calculate_timestep();

    // 5. 计算粘性残差
    if (config_->physics.viscous_mode == 1) {
        calculate_viscous_rhs();
        communicate_dq_npp();
    }

    // 6. 计算无粘残差
    calculate_inviscous_rhs();

    communicate_dq_npp();

    // 7. 时间推进部分
    time_step_lhs_tgh();

    communicate_pv_npp();
}

// =========================================================================
// 残差监测：计算 RMS 和 Max Residual (修正版：单行极简输出)
// =========================================================================
void SolverDriver::check_residual(int step_idx) {
    int rank = mpi_.get_rank();
    const int nl = 5;
    
    // 配置参数 (对应 Fortran m1 = 1 + method)
    const int m1 = 2; 

    // 本地统计变量
    double local_sum_sq = 0.0;
    long long local_count = 0;
    
    struct MaxResInfo {
        double val;
        int rank;
    } local_max_res, global_max_res;

    local_max_res.val = -1.0;
    local_max_res.rank = rank;

    // 记录最大值的位置信息
    struct LocInfo {
        int bid;
        int i, j, k, m;
    } local_loc, global_loc;
    
    // 初始化位置
    local_loc = {0, 0, 0, 0, 0};

    // -----------------------------------------------------
    // 1. 遍历本地 Block 进行统计 (保持不变)
    // -----------------------------------------------------
    for (auto* b : blocks_) {
        if (!b || b->owner_rank != rank) continue;

        int ni = b->ni;
        int nj = b->nj;
        int nk = b->nk;
        int ng = b->ng;

        int i_start = m1 - 1; 
        int j_start = m1 - 1;
        int k_start = m1 - 1;
        
        int i_end = ni - 2;
        int j_end = nj - 2;
        int k_end = nk - 2;

        if (i_start > i_end || j_start > j_end || k_start > k_end) continue;

        for (int k = k_start; k <= k_end; ++k) {
            for (int j = j_start; j <= j_end; ++j) {
                for (int i = i_start; i <= i_end; ++i) {
                    
                    int I = i + ng;
                    int J = j + ng;
                    int K = k + ng;

                    double dt = b->dtdt(I, J, K);
                    double vol = b->vol(I, J, K);
                    
                    if (dt <= 1.0e-30) dt = 1.0e-30;
                    if (vol <= 1.0e-30) vol = 1.0e-30;

                    double dtime1 = 1.0 / (dt * vol);

                    for (int m = 0; m < nl; ++m) {
                        double dq0 = std::abs(b->dq(I, J, K, m));
                        double dresm = dq0 * dtime1;

                        local_sum_sq += dresm * dresm;
                        
                        if (dresm > local_max_res.val) {
                            local_max_res.val = dresm;
                            local_loc.bid = b->id;
                            // 记录物理索引 (1-based, 方便与 Fortran 对比)
                            local_loc.i = i + 1; 
                            local_loc.j = j + 1; 
                            local_loc.k = k + 1;
                            local_loc.m = m + 1; 
                        }
                    }
                    local_count += nl;
                }
            }
        }
    }

    // -----------------------------------------------------
    // 2. MPI 全局规约 (保持不变)
    // -----------------------------------------------------
    double global_sum_sq = 0.0;
    long long global_count = 0;

    MPI_Allreduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max_res, &global_max_res, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    // -----------------------------------------------------
    // 3. 广播最大值位置详情 (保持不变)
    // -----------------------------------------------------
    int loc_buf[5] = {local_loc.bid, local_loc.i, local_loc.j, local_loc.k, local_loc.m};
    
    if (rank != global_max_res.rank) {
        loc_buf[0] = 0;
    }
    
    MPI_Bcast(loc_buf, 5, MPI_INT, global_max_res.rank, MPI_COMM_WORLD);
    
    global_loc.bid = loc_buf[0];
    global_loc.i   = loc_buf[1];
    global_loc.j   = loc_buf[2];
    global_loc.k   = loc_buf[3];
    global_loc.m   = loc_buf[4];

    // -----------------------------------------------------
    // 4. 计算并打印结果 (修改：单行输出逻辑)
    // -----------------------------------------------------
    if (rank == 0) {
        double rms_residual = 0.0;
        if (global_count > 0) {
            rms_residual = std::sqrt(global_sum_sq / double(global_count));
        }

        // 使用静态变量控制表头只在当前运行的“第一次”调用时打印
        // 这样无论是 Step 1 还是 Restart 的 Step 100，第一次都会显示表头
        static bool is_first_print = true;
        
        if (is_first_print) {
            // 打印表头：Step, RMS, Max, Location Info
            spdlog::info("{:>8}  {:>14}  {:>14}  {:^30}", 
                         "Step", "RMS Residual", "Max Residual", "Max Loc [Blk ( i, j, k) #Var]");
            is_first_print = false;
        }

        // 格式化位置字符串: "62 (  3,   2,  12) #5"
        std::string loc_str = fmt::format("{:<4} ({:>3}, {:>3}, {:>3}) #{:<1}", 
                                          global_loc.bid, global_loc.i, global_loc.j, global_loc.k, global_loc.m);

        // 单行输出数据
        spdlog::info("{:8d}  {:14.6E}  {:14.6E}  {}", 
                     step_idx, rms_residual, global_max_res.val, loc_str);
    }
}

void SolverDriver::check_io_output(int step) {
    // TODO: 实现文件保存
    if (mpi_.is_root()) {
        spdlog::info("Step {}: Saving solution... (TODO)", step);
    }
}

// =========================================================================
// 核心函数：exchange_bc (修正接口调用)
// =========================================================================
void SolverDriver::exchange_bc() {
    int rank = mpi_.get_rank();
    int iseq = 0; // 全局 Tag 计数器
    
    const int mcyc[5] = {0, 1, 2, 0, 1};
    int nblocks = blocks_.size();

    // 1. 全局遍历所有块、所有边界 (保持 Tag 同步)
    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b_meta = blocks_[nb]; 
        
        // 即使是非本地块，我们在 pre 阶段也保留了 shell (边界信息)，所以不应为空
        if (!b_meta) {
            spdlog::critical("Block {} is null! Tag synchronization will fail.", nb+1);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int id_src = b_meta->owner_rank;
        int nregions = b_meta->boundaries.size();

        for (int nr = 0; nr < nregions; ++nr) {
            BoundaryPatch& patch = b_meta->boundaries[nr];
            int ibctype = patch.type;

            // 仅处理对接边界 (Type < 0)
            if (ibctype < 0) {
                int nbt = patch.target_block; // 0-based index
                
                if (blocks_[nbt] == nullptr) {
                     spdlog::critical("Target Block {} is null!", nbt+1);
                     MPI_Abort(MPI_COMM_WORLD, 1);
                }
                int id_des = blocks_[nbt]->owner_rank;

                // 仅处理跨进程通信
                if (id_src != id_des) {
                    iseq++; 
                    int tag_seq = iseq;
                    
                    int nrt = patch.window_id - 1; // 1-based -> 0-based

                    // --- 几何范围计算 ---
                    int ibeg[3], iend[3];
                    for(int k=0; k<3; ++k) {
                        ibeg[k] = std::abs(patch.raw_is[k]);
                        iend[k] = std::abs(patch.raw_ie[k]);
                    }
                    
                    int idir = patch.s_nd; 
                    int inrout = patch.s_lr; 

                    ibeg[idir] -= 4 * std::max(inrout, 0);
                    iend[idir] -= 4 * std::min(inrout, 0);

                    for (int m = 0; m < 2; ++m) {
                        int idir2 = mcyc[idir + (m + 1)]; 
                        if (ibeg[idir2] > 1) ibeg[idir2] -= 1;
                        
                        int dim_limit = 0;
                        if (idir2 == 0) dim_limit = b_meta->ni;
                        else if (idir2 == 1) dim_limit = b_meta->nj;
                        else dim_limit = b_meta->nk;

                        if (iend[idir2] < dim_limit) iend[idir2] += 1;
                    }

                    int size_i = iend[0] - ibeg[0] + 1;
                    int size_j = iend[1] - ibeg[1] + 1;
                    int size_k = iend[2] - ibeg[2] + 1;
                    size_t packsize = (size_t)size_i * size_j * size_k * 5;

                    // ==========================================
                    // 发送方逻辑 (Sender)
                    // ==========================================
                    if (rank == id_src) {
                        Block* b = b_meta; 
                        auto& qpv = patch.qpv; 
                        int ng = b->ng;

                        // 打包数据
                        if (packsize > 0) {
                            // qpv 在 pre 阶段应该已经 resize 过了
                            // 遍历 1-based 范围，映射到 buffer 的 0-based
                            for (int k = ibeg[2]; k <= iend[2]; ++k) {
                                for (int j = ibeg[1]; j <= iend[1]; ++j) {
                                    for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                        // 物理场索引 (0-based + ghost)
                                        int ix = (i - 1) + ng;
                                        int iy = (j - 1) + ng;
                                        int iz = (k - 1) + ng;

                                        // Buffer 索引 (0-based)
                                        int bi = i - ibeg[0];
                                        int bj = j - ibeg[1];
                                        int bk = k - ibeg[2];

                                        qpv(bi, bj, bk, 0) = b->r(ix, iy, iz);
                                        qpv(bi, bj, bk, 1) = b->u(ix, iy, iz);
                                        qpv(bi, bj, bk, 2) = b->v(ix, iy, iz);
                                        qpv(bi, bj, bk, 3) = b->w(ix, iy, iz);
                                        qpv(bi, bj, bk, 4) = b->p(ix, iy, iz);
                                    }
                                }
                            }
                            
                            // 【修正】使用 .host_data() 获取底层指针
                            MPI_Send(qpv.host_data(), packsize, MPI_DOUBLE, id_des, tag_seq, MPI_COMM_WORLD);
                        }
                    }

                    // ==========================================
                    // 接收方逻辑 (Receiver)
                    // ==========================================
                    if (rank == id_des) {
                        Block* b_target = blocks_[nbt];
                        BoundaryPatch& target_patch = b_target->boundaries[nrt];
                        auto& qpvpack = target_patch.qpvpack;

                        if (packsize > 0) {
                            // 【修正】使用 .host_data() 获取底层指针
                            MPI_Recv(qpvpack.host_data(), packsize, MPI_DOUBLE, id_src, tag_seq, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                    }

                } // end if cross-process
            } // end if type < 0
        } // end nr
    } // end nb
}

// =========================================================================
// 驱动：boundary_all
// =========================================================================
void SolverDriver::boundary_all() {
    int rank = mpi_.get_rank();
    for (Block* b : blocks_) {
        if (b && b->owner_rank == rank) {
            boundary_manager_.boundary_sequence(b);
        }
    }
}

// =========================================================================
// 核心驱动：物理场状态同步与初始化
// 作用：在计算 dt 或 Flux 之前，确保 c, T, mu, Q 与 rho, u, v, w, p 一致
// =========================================================================
void SolverDriver::update_derived_variables() {
    
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
// =========================================================================
// 核心计算：更新声速
// 仅计算物理区和紧邻的第 1 层 Ghost (Face Neighbors)，严格跳过角点
// =========================================================================
void SolverDriver::update_sound_speed(Block* b) {
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

// =========================================================================
// 核心计算：更新守恒变量 Q (对应 get_q_only)
// 逻辑：Primitive (rho, u, v, w, p) -> Conservative (rho, rhou, rhov, rhow, rhoE)
// 特性：合并了物理区和 Ghost 区的处理，支持负密度 Flag
// =========================================================================
void SolverDriver::update_conservative_vars(Block* b) {
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

// =========================================================================
// 核心工具：清空残差/增量数组 (对应 set_dq_to_0)
// =========================================================================
void SolverDriver::reset_residuals(Block* b) {
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

// =========================================================================
// 核心计算：更新热力学变量 (严格复刻 Fortran get_c_t_only)
// 逻辑：
//    1. 使用 "Strip Mining" (条带式) 循环，严格避开角点 (Corners)
//    2. 只有物理区和 n1 层计算声速 c
//    3. n1, n2, n3 层都计算温度 t
//    4. 只有 n3 层取绝对值 abs(rho)
// =========================================================================
void SolverDriver::update_thermodynamics(Block* b) {
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

// =========================================================================
// 核心计算：更新层流粘性系数
// 公式：visl = tm * sqrt(tm) * (1.0 + visc) / (tm + visc)
// =========================================================================
void SolverDriver::update_laminar_viscosity(Block* b) {
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

// =========================================================================
// 核心驱动：计算时间步长
// 逻辑流程：
//    1. 遍历所有块：计算谱半径 -> 计算局部 dt
//    2. 根据 ntmst 模式统计或修正 dt
// =========================================================================
void SolverDriver::calculate_timestep() {
    auto& global = GlobalData::getInstance();
    int ntmst = config_->timeStep.mode; 
    int rank = mpi_.get_rank();
    
    real_t dtmin = 1.0e12; // 初始值

    // [验证准备] 初始化本地极值变量
    real_t local_min_dtdt = 1.0e30;
    real_t local_max_dtdt = -1.0e30;

    // --------------------------------------------------------
    // 第一阶段：逐块计算 (Loop 1)
    // --------------------------------------------------------
    for (Block* b : blocks_) {
        if (b && b->owner_rank == rank) {
            spectrum_tgh(b);

            localdt0(b);
        }
    }
}

// =========================================================================
// 核心计算：计算总谱半径
// 逻辑：
//    1. 调用 spectinv 计算对流谱半径
//    2. 如果是粘性流 (nvis=1) 且 csrv > 0，则计算粘性谱半径 (spectvisl)
//    3. 否则将粘性谱半径清零
// =========================================================================
void SolverDriver::spectrum_tgh(Block* b) {
    // 1. 计算对流谱半径 (Inviscid Spectral Radius)
    // 对应 Fortran: call spectinv
    spectinv(b); 

    // 2. 处理粘性谱半径 (Viscous Spectral Radius)
    int nvis = config_->physics.viscous_mode;
    real_t csrv = config_->scheme.visc_spectral_radius;

    if (nvis == 1) {
        if (csrv > 0.0) {
            spectvisl(b);
        }
    } else {
        set_spectvis_to_0(b);
    }
}

// =========================================================================
// 工具函数：清空粘性谱半径
// 作用：将 srva, srvb, srvc 全场置为 0
// =========================================================================
void SolverDriver::set_spectvis_to_0(Block* b) {
    // C++ 逻辑：为了安全和一致性，我们清空整个分配的内存块 (0 到 total_size)
    
    int size_i = b->ni + 2 * b->ng;
    int size_j = b->nj + 2 * b->ng;
    int size_k = b->nk + 2 * b->ng;

    for (int k = 0; k < size_k; ++k) {
        for (int j = 0; j < size_j; ++j) {
            for (int i = 0; i < size_i; ++i) {
                b->srva(i, j, k) = 0.0;
                b->srvb(i, j, k) = 0.0;
                b->srvc(i, j, k) = 0.0;
            }
        }
    }
}

// =========================================================================
// 核心计算：对流谱半径
// 逻辑：
//    1. 计算物理区 (Inner) 的 sra, srb, src
//    2. 计算 Ghost 边界 (I/J/K faces) 的 sra, srb, src
//       注意：边界计算时，度量(Metric)取自最近的物理网格(Clamped)，
//             但流场变量(U,V,W,C)取自Ghost网格本身。
// =========================================================================
void SolverDriver::spectinv(Block* b) {
    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 辅助 Lambda：计算单点的谱半径
    // 参数：
    //   i, j, k: 当前网格坐标 (用于读取 U, V, W, C 和 写入 sra, srb, src)
    //   mi, mj, mk: 度量坐标 (Metric Index, 用于读取 kcx, etx 等)
    //               在物理区 mi=i, mj=j, mk=k；在边界处会被钳位(Clamp)到物理边界
    auto calc_point_spect = [&](int i, int j, int k, int mi, int mj, int mk) {
        real_t u = b->u(i, j, k);
        real_t v = b->v(i, j, k);
        real_t w = b->w(i, j, k);
        real_t c = b->c(i, j, k);

        // 1. 获取度量 (对应 getr0kec_mml 的三次调用)
        // Xi metrics (from mi, mj, mk)
        real_t kx = b->kcx(mi, mj, mk);
        real_t ky = b->kcy(mi, mj, mk);
        real_t kz = b->kcz(mi, mj, mk);
        real_t kt = b->kct(mi, mj, mk);

        // Eta metrics
        real_t ex = b->etx(mi, mj, mk);
        real_t ey = b->ety(mi, mj, mk);
        real_t ez = b->etz(mi, mj, mk);
        real_t et = b->ett(mi, mj, mk);

        // Zeta metrics
        real_t cx = b->ctx(mi, mj, mk);
        real_t cy = b->cty(mi, mj, mk);
        real_t cz = b->ctz(mi, mj, mk);
        real_t ct = b->ctt(mi, mj, mk);

        // 2. 计算逆变速度分量 (Contravariant Velocities)
        real_t cita = kx*u + ky*v + kz*w + kt;
        real_t citb = ex*u + ey*v + ez*w + et;
        real_t citc = cx*u + cy*v + cz*w + ct;

        // 3. 计算梯度模长 (Gradient Magnitudes)
        real_t cgma = std::sqrt(kx*kx + ky*ky + kz*kz);
        real_t cgmb = std::sqrt(ex*ex + ey*ey + ez*ez);
        real_t cgmc = std::sqrt(cx*cx + cy*cy + cz*cz);

        // 4. 赋值谱半径
        b->sra(i, j, k) = std::abs(cita) + c * cgma;
        b->srb(i, j, k) = std::abs(citb) + c * cgmb;
        b->src(i, j, k) = std::abs(citc) + c * cgmc;
    };

    // ========================================================
    // 1. 物理核心区域 (Inner Loop)
    // ========================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                // 物理区：度量坐标 = 网格坐标
                calc_point_spect(I, J, K, I, J, K);
            }
        }
    }

    // ========================================================
    // 2. I 方向边界 (I-Boundaries)
    // ii = max(1, min(ni, i)) -> 钳位逻辑
    // ========================================================
    int i_bounds[] = {ng - 1, ni + ng}; // 对应 Fortran 0 和 ni+1
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int I : i_bounds) {
                int J = j + ng;
                int K = k + ng;
                
                // 钳位逻辑 (Clamping)
                // 如果 I 是左 Ghost (ng-1)，取物理左边界 (ng)
                // 如果 I 是右 Ghost (ni+ng)，取物理右边界 (ni+ng-1)
                int mi = I;
                if (mi < ng) mi = ng;
                if (mi >= ni + ng) mi = ni + ng - 1;

                calc_point_spect(I, J, K, mi, J, K);
            }
        }
    }

    // ========================================================
    // 3. J 方向边界 (J-Boundaries)
    // ========================================================
    int j_bounds[] = {ng - 1, nj + ng}; 
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            for (int J : j_bounds) {
                int I = i + ng;
                int K = k + ng;

                int mj = J;
                if (mj < ng) mj = ng;
                if (mj >= nj + ng) mj = nj + ng - 1;

                calc_point_spect(I, J, K, I, mj, K);
            }
        }
    }

    // ========================================================
    // 4. K 方向边界 (K-Boundaries)
    // ========================================================
    int k_bounds[] = {ng - 1, nk + ng};
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            for (int K : k_bounds) {
                int I = i + ng;
                int J = j + ng;

                int mk = K;
                if (mk < ng) mk = ng;
                if (mk >= nk + ng) mk = nk + ng - 1;

                calc_point_spect(I, J, K, I, J, mk);
            }
        }
    }
}

// =========================================================================
// 核心计算：粘性谱半径
// 逻辑：
//    1. 计算物理区 (Inner) 的 srva, srvb, srvc
//    2. 计算 Ghost 边界 (I/J/K faces) 的 srva, srvb, srvc
//    注意：为了 100% 复刻 Fortran，边界处的 visl、vol 和 Metrics 
//          都取自最近的物理网格 (Clamped Index)，而坐标变量仅用于写入。
// =========================================================================
void SolverDriver::spectvisl(Block* b) {
    const auto& global = GlobalData::getInstance();
    real_t csrv = config_->scheme.visc_spectral_radius;
    real_t reynolds = global.reynolds;
    real_t small = 1.0e-38; 

    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 辅助 Lambda：计算单点的粘性谱半径
    // 参数：
    //   i, j, k: 目标写入坐标 (srva, srvb, srvc)
    //   mi, mj, mk: 源数据读取坐标 (vol, visl, Metrics) - 已钳位到物理区
    auto calc_visc_spect = [&](int i, int j, int k, int mi, int mj, int mk) {
        
        // 1. 获取几何与物理量 (全部来自 Clamped Index mi,mj,mk)
        // 对应 Fortran: rm_vol = r(i,j,k) * vol(ii,j,k) ? 
        // 等等！Fortran 代码里 inner loop 是 r(i,j,k)*vol(i,j,k)
        // 但在边界循环里：
        //    rm_vol = r(i,j,k) * vol(ii,j,k)  <-- 注意！r 用的是 i (Ghost), vol 用的是 ii (Clamped)
        //    vis = visl(ii,j,k)               <-- vis 用的是 ii (Clamped)
        
        // 我们必须严格区分：
        //   - r (密度) : 使用原始坐标 i,j,k
        //   - vol, visl, Metrics : 使用钳位坐标 mi,mj,mk
        
        real_t rho = b->r(i, j, k);       // Raw index
        real_t vol = b->vol(mi, mj, mk);  // Clamped index
        real_t vis = b->visl(mi, mj, mk); // Clamped index

        real_t rm_vol = rho * vol; 

        // 2. 获取度量 (Clamped Metrics)
        real_t kx = b->kcx(mi, mj, mk);
        real_t ky = b->kcy(mi, mj, mk);
        real_t kz = b->kcz(mi, mj, mk);
        
        real_t ex = b->etx(mi, mj, mk);
        real_t ey = b->ety(mi, mj, mk);
        real_t ez = b->etz(mi, mj, mk);
        
        real_t cx = b->ctx(mi, mj, mk);
        real_t cy = b->cty(mi, mj, mk);
        real_t cz = b->ctz(mi, mj, mk);

        // 3. 计算度量平方和 (ri, rj, rk)
        real_t ri = kx*kx + ky*ky + kz*kz;
        real_t rj = ex*ex + ey*ey + ez*ez;
        real_t rk = cx*cx + cy*cy + cz*cz;

        // 4. 计算系数
        // coef = 2.0 * vis / (reynolds * rm_vol + small)
        real_t denom = reynolds * rm_vol + small;
        real_t coef = 2.0 * vis / denom;
        real_t coef1 = csrv * coef;

        // 5. 赋值粘性谱半径
        b->srva(i, j, k) = coef1 * ri;
        b->srvb(i, j, k) = coef1 * rj;
        b->srvc(i, j, k) = coef1 * rk;
    };

    // ========================================================
    // 1. 物理核心区域 (Inner Loop)
    // 对应 Fortran: do k=1,nk; do j=1,nj; do i=1,ni
    // ========================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                // 物理区：读取坐标 = 写入坐标
                calc_visc_spect(I, J, K, I, J, K);
            }
        }
    }

    // ========================================================
    // 2. I 方向边界 (I-Boundaries)
    // 对应 Fortran: do i=0, ni+1, ni+1
    // ii = max(1, min(ni, i))
    // ========================================================
    int i_bounds[] = {ng - 1, ni + ng}; 
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int I : i_bounds) {
                int J = j + ng;
                int K = k + ng;
                
                // Clamping
                int mi = I;
                if (mi < ng) mi = ng;
                if (mi >= ni + ng) mi = ni + ng - 1;

                calc_visc_spect(I, J, K, mi, J, K);
            }
        }
    }

    // ========================================================
    // 3. J 方向边界 (J-Boundaries)
    // 对应 Fortran: do j=0, nj+1, nj+1
    // ========================================================
    int j_bounds[] = {ng - 1, nj + ng};
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            for (int J : j_bounds) {
                int I = i + ng;
                int K = k + ng;

                int mj = J;
                if (mj < ng) mj = ng;
                if (mj >= nj + ng) mj = nj + ng - 1;

                calc_visc_spect(I, J, K, I, mj, K);
            }
        }
    }

    // ========================================================
    // 4. K 方向边界 (K-Boundaries)
    // 对应 Fortran: do k=0, nk+1, nk+1
    // ========================================================
    int k_bounds[] = {ng - 1, nk + ng};
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            for (int K : k_bounds) {
                int I = i + ng;
                int J = j + ng;

                int mk = K;
                if (mk < ng) mk = ng;
                if (mk >= nk + ng) mk = nk + ng - 1;

                calc_visc_spect(I, J, K, I, J, mk);
            }
        }
    }
}

// =========================================================================
// 核心计算：计算局部时间步长
// =========================================================================
void SolverDriver::localdt0(Block* b) {
    auto& global = GlobalData::getInstance();
    
    // 配置参数
    real_t cfl = config_->timeStep.cfl;
    real_t timedt_rate = config_->timeStep.dt_rate;
    real_t small = 1.0e-38;

    global.timedt = small;

    int ng = b->ng;
    
    // 对应 C++ 物理区: ng 到 ni+ng
    int is = ng; 
    int ie = b->ni + ng;
    
    int js = ng; 
    int je = b->nj + ng;
    
    int ks = ng; 
    int ke = b->nk + ng;

    for (int k = ks; k < ke; ++k) {
        for (int j = js; j < je; ++j) {
            for (int i = is; i < ie; ++i) {
                
                real_t ra = b->sra(i, j, k);
                real_t rb = b->srb(i, j, k);
                real_t rc = b->src(i, j, k);
                
                // 粘性谱半径求和
                real_t rv = b->srva(i, j, k) + b->srvb(i, j, k) + b->srvc(i, j, k);

                // 总谱半径 rabc
                real_t rabc = ra + rb + rc + rv;

                // 2. 计算时间步长
                real_t dt_val;
                
                if (timedt_rate > small) {
                    real_t vol = b->vol(i, j, k);
                    // Fortran: min(cfl/rabc, timedt_rate/vol)
                    real_t dt1 = cfl / rabc;
                    real_t dt2 = timedt_rate / vol;
                    dt_val = (dt1 < dt2) ? dt1 : dt2;
                } else {
                    dt_val = cfl / rabc;
                }

                // 3. 存入 dtdt 数组
                b->dtdt(i, j, k) = dt_val;

                // 4. 更新 timedt (统计 max(dt * vol))
                real_t val = dt_val * b->vol(i, j, k);
                if (val > global.timedt) {
                    global.timedt = val;
                }
            }
        }
    }
}

// =========================================================================
// 驱动函数：计算粘性右端项 (原 r_h_s_vis)
// =========================================================================
void SolverDriver::calculate_viscous_rhs() {
    int rank = mpi_.get_rank();

    for (Block* b : blocks_) {
        // 1. 检查是否为本地 Block
        if (b && b->owner_rank == rank) {
            
            // 2. 调用核心计算逻辑 (针对单个 Block)
            compute_block_viscous_rhs(b);
        }
    }
}

// =========================================================================
// 核心计算：单个 Block 的粘性计算
// 对应 Fortran: subroutine WCNSE5_VIS_virtual
// =========================================================================
void SolverDriver::compute_block_viscous_rhs(Block* b) {
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

// =========================================================================
// 计算节点上的导数 (对应 Fortran: subroutine UVWT_DER_4th_virtual)
// =========================================================================
void SolverDriver::uvwt_der_4th_virtual(Block* b, HyresArray<real_t>& duvwt) {
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

// =========================================================================
// 一维求导核心算法 (修正版：严格复刻 src-fortran/newvis.f90 DUVWT_NODE_LINE_virtual)
// 注意：这是计算节点导数 (Node Derivative)，用于 u, v, w, T
// =========================================================================
void SolverDriver::compute_duvwt_node_line_virtual(
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
// 计算半点/界面上的导数 (对应 Fortran: subroutine UVWT_DER_4th_half_virtual)
// 结果存入 duvwt_mid (I:0-3, J:4-7, K:8-11)
// =========================================================================
void SolverDriver::uvwt_der_4th_half_virtual(Block* b, HyresArray<real_t>& duvwt_mid) {
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
void SolverDriver::compute_duvwt_half_line_virtual(
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
void SolverDriver::compute_value_line_half_virtual(
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
void SolverDriver::compute_value_half_node_ijk(
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
void SolverDriver::compute_value_half_node(
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
void SolverDriver::compute_duvwt_dxyz(
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
void SolverDriver::compute_flux_vis_line_new(
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
// 计算通量导数 (修正版：修复索引偏移错误)
// =========================================================================
void SolverDriver::compute_flux_dxyz(
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

// =========================================================================
// 标量半点插值 (新增：用于处理一维 vector)
// =========================================================================
void SolverDriver::compute_value_half_node_scalar(
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
// 全局 DQ 通信与归一化 (修复索引排序问题)
// =========================================================================
void SolverDriver::communicate_dq_npp() {
    // 1. 交换边界残差与体积 (已实现)
    // 对应 Fortran: call exchange_bc_dq_vol
    exchange_bc_dq_vol();

    // ---------------------------------------------------------------------
    // Part A: 收集本地奇异点数据到发送缓冲区 (Gather Local)
    // 对应 Fortran loop: do i=n,m ... dq_npp_local(j,i) = ...
    // ---------------------------------------------------------------------
    auto& global = GlobalData::getInstance();
    int rank = mpi_.get_rank();
    int n_vars = 5;

    // 遍历属于当前 Rank 的所有 Block
    for (Block* b : blocks_) {
        // 严格检查归属权 (只处理本地 Block)
        if (!b || b->owner_rank != rank) continue;

        int ng = b->ng;

        // 遍历该 Block 上的所有本地奇异点
        // (Block::LocalSingularPoint 已经在 build_nppos_list_ex 中建立)
        for (const auto& p : b->singular_points) {
            // p.i, p.j, p.k 是 0-based 物理坐标，需要加上 ghost 偏移
            int I = p.i + ng;
            int J = p.j + ng;
            int K = p.k + ng;

            real_t vol_p = b->vol(I, J, K);
            // 保护一下除零 (虽然物理网格体积不应为0)
            if (vol_p < 1.0e-30) vol_p = 1.0e-30; 

            // p.buffer_seq 是该点在本地发送 buffer 中的 0-based 序列号
            // 对应 Fortran: i - n (相对偏移)
            int base_idx = p.buffer_seq * n_vars;

            for (int m = 0; m < n_vars; ++m) {
                // Fortran: dq_npp_local(j, i) = mb_dq(...)/vol
                // C++ Buffer 布局为 Flat [point][var]
                global.dq_npp_local[base_idx + m] = b->dq(I, J, K, m) / vol_p;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Part B: MPI 全局通信 (Allgatherv)
    // 对应 Fortran: call MPI_ALLGATHERV(...)
    // ---------------------------------------------------------------------
    int num_procs = mpi_.get_size();
    
    // 准备 MPI 参数 (单位: double)
    // 注意: global.nppos_local 和 ipp_st_local 存储的是点数，需要乘以 5 (变量数)
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);

    for (int r = 0; r < num_procs; ++r) {
        recvcounts[r] = global.nppos_local[r] * n_vars;
        displs[r]     = global.ipp_st_local[r] * n_vars;
    }

    // 本地发送数据量
    int sendcount = global.nppos_local[rank] * n_vars;

    // 执行通信
    MPI_Allgatherv(global.dq_npp_local.data(), sendcount, MPI_DOUBLE,
                   global.dq_npp_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // 2. 对接面平滑 (2 Point Match) (已实现)
    // 对应 Fortran: call boundary_match_dq_2pm
    boundary_match_dq_2pm();

    // 3. 多点平均 (3 Point Plus)
    // 对应 Fortran: call boundary_match_dq_3pp
    boundary_match_dq_3pp();
}

// =========================================================================
// 多点平均处理子函数 (Strictly following Fortran logic)
// 对应 Fortran: subroutine boundary_match_dq_3pp
// =========================================================================
void SolverDriver::boundary_match_dq_3pp() {
    auto& global = GlobalData::getInstance();
    int rank = mpi_.get_rank();
    int n_vars = 5;

    // Fortran 逻辑是遍历本地负责的 index，然后找对应的 group。
    // C++ 中由于数据结构差异 (GlobalData 中直接存储了 singularity_groups)，
    // 我们直接遍历所有 SingularityGroup 是等价且更自然的。
    // 只要最后只更新 "属于本地的 Block" 的值，结果就是一致的。

    for (const auto& group : global.singularity_groups) {
        int npp = group.points.size();
        
        // 1. 计算该组的平均值 (Sum)
        std::array<real_t, 5> dqq = {0.0, 0.0, 0.0, 0.0, 0.0};

        for (const auto& p : group.points) {
            // p.global_buffer_index 是该点在全局 buffer 中的绝对索引 (0-based)
            int base_idx = p.global_buffer_index * n_vars;
            
            for (int m = 0; m < n_vars; ++m) {
                dqq[m] += global.dq_npp_global[base_idx + m];
            }
        }

        // 2. 求平均 (Divide)
        real_t inv_npp = 1.0 / static_cast<real_t>(npp);
        for (int m = 0; m < n_vars; ++m) {
            dqq[m] *= inv_npp;
        }

        // 3. 将平均值回填到该组包含的所有点 (Update)
        for (const auto& p : group.points) {
            // 获取对应的 Block 指针
            // 此时 p.block_id 是全局 Block ID，需要映射到本地 blocks_ 容器
            // 假设 blocks_ 容器中，非本地 Block 为 nullptr 或者只包含 Shell。
            // 关键：我们只需要更新 "Owner 是我" 的 Block。
            
            // 注意：blocks_ 向量通常按 ID 排序。如果 blocks_ 是全局大小，可以直接索引。
            // 如果 blocks_ 只存本地块，则需要查找。
            // 根据之前的 SolverDriver 构造逻辑，blocks_ 应该是包含所有 Block 指针的 vector (大小=n_blocks)，
            // 其中非本地的可能是 nullptr 或者 ghost shell。
            
            if (p.block_id >= blocks_.size()) continue; // 安全检查
            Block* b = blocks_[p.block_id];

            // 仅更新属于当前进程的 Block
            if (b && b->owner_rank == rank) {
                int ng = b->ng;
                // p.i, p.j, p.k 是 0-based 物理坐标 -> 转为内存坐标 (+ng)
                int I = p.i + ng;
                int J = p.j + ng;
                int K = p.k + ng;

                real_t vol_p = b->vol(I, J, K);

                for (int m = 0; m < n_vars; ++m) {
                    // Fortran: mb_dq(nb)%a4d(...) = dqq(m) * vol_p
                    b->dq(I, J, K, m) = dqq[m] * vol_p;
                }
            }
        }
    }
}

// =========================================================================
// 全局 PV (原始变量) 通信与归一化
// 对应 Fortran: subroutine communicate_pv_npp
// =========================================================================
void SolverDriver::communicate_pv_npp() {
    // 1. 交换边界原始变量 (rho, u, v, w, p, t) 与 体积 (已实现)
    exchange_bc_pv_vol();

    // ---------------------------------------------------------------------
    // Part A: 收集本地奇异点数据到发送缓冲区 (Gather Local PV)
    // 对应 Fortran loop: do i=n,m ... pv_npp_local(j,i) = ...
    // ---------------------------------------------------------------------
    auto& global = GlobalData::getInstance();
    int rank = mpi_.get_rank();
    int n_vars = 6; // r, u, v, w, p, t

    // 遍历属于当前 Rank 的所有 Block
    for (Block* b : blocks_) {
        if (!b || b->owner_rank != rank) continue;

        int ng = b->ng;

        // 遍历该 Block 上的所有本地奇异点
        for (const auto& p : b->singular_points) {
            int I = p.i + ng;
            int J = p.j + ng;
            int K = p.k + ng;

            // 【关键点】Fortran 代码中 vol_p = 1.0 (!!mb_vol 被注释)
            // 原始变量直接平均，不进行体积加权
            real_t vol_p = 1.0; 

            // p.buffer_seq 是该点在本地发送 buffer 中的 0-based 序列号
            int base_idx = p.buffer_seq * n_vars;

            // 填充 pv_npp_local (6 vars)
            // 对应 Fortran: mb_r/vol_p, mb_u/vol_p ...
            global.pv_npp_local[base_idx + 0] = b->r(I, J, K) / vol_p;
            global.pv_npp_local[base_idx + 1] = b->u(I, J, K) / vol_p;
            global.pv_npp_local[base_idx + 2] = b->v(I, J, K) / vol_p;
            global.pv_npp_local[base_idx + 3] = b->w(I, J, K) / vol_p;
            global.pv_npp_local[base_idx + 4] = b->p(I, J, K) / vol_p;
            global.pv_npp_local[base_idx + 5] = b->t(I, J, K) / vol_p;
        }
    }

    // ---------------------------------------------------------------------
    // Part B: MPI 全局通信 (Allgatherv)
    // 对应 Fortran: call MPI_ALLGATHERV(..., pv_npp, ...)
    // ---------------------------------------------------------------------
    int num_procs = mpi_.get_size();
    
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);

    for (int r = 0; r < num_procs; ++r) {
        recvcounts[r] = global.nppos_local[r] * n_vars; // *6
        displs[r]     = global.ipp_st_local[r] * n_vars; // *6
    }

    int sendcount = global.nppos_local[rank] * n_vars;

    // 执行通信 (注意 buffer 是 pv_npp 系列)
    MPI_Allgatherv(global.pv_npp_local.data(), sendcount, MPI_DOUBLE,
                   global.pv_npp_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // 2. 对接面平滑 (2 Point Match) (已实现)
    boundary_match_pv_2pm();

    // 3. 多点平均 (3 Point Plus) (新增)
    boundary_match_pv_3pp();
}

// =========================================================================
// 原始变量多点平均 (Strictly following Fortran logic)
// 对应 Fortran: subroutine boundary_match_pv_3pp
// =========================================================================
void SolverDriver::boundary_match_pv_3pp() {
    auto& global = GlobalData::getInstance();
    int rank = mpi_.get_rank();
    int n_vars = 6; // r, u, v, w, p, t

    // 遍历所有奇异点组 (全局拓扑)
    // 逻辑：计算组平均值 -> 回填到本地 Block
    for (const auto& group : global.singularity_groups) {
        int npp = group.points.size();
        
        // 1. 计算该组的平均值 (Sum)
        std::array<real_t, 6> dqq = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        for (const auto& p : group.points) {
            // p.global_buffer_index 是该点在全局 buffer 中的绝对索引
            int base_idx = p.global_buffer_index * n_vars;
            
            for (int m = 0; m < n_vars; ++m) {
                dqq[m] += global.pv_npp_global[base_idx + m];
            }
        }

        // 2. 求平均 (Divide)
        real_t inv_npp = 1.0 / static_cast<real_t>(npp);
        for (int m = 0; m < n_vars; ++m) {
            dqq[m] *= inv_npp;
        }

        // 3. 将平均值回填到该组包含的所有点 (Update Local Blocks)
        for (const auto& p : group.points) {
            if (p.block_id >= blocks_.size()) continue; 
            Block* b = blocks_[p.block_id];

            // 仅更新属于当前进程的 Block
            if (b && b->owner_rank == rank) {
                int ng = b->ng;
                int I = p.i + ng;
                int J = p.j + ng;
                int K = p.k + ng;

                // 【关键点】Fortran 代码中 vol_p = 1.0
                real_t vol_p = 1.0;

                // 回填 6 个变量
                b->r(I, J, K) = dqq[0] * vol_p;
                b->u(I, J, K) = dqq[1] * vol_p;
                b->v(I, J, K) = dqq[2] * vol_p;
                b->w(I, J, K) = dqq[3] * vol_p;
                b->p(I, J, K) = dqq[4] * vol_p;
                b->t(I, J, K) = dqq[5] * vol_p;
            }
        }
    }
}

// =========================================================================
// 交换边界 DQ 和 Volume (修复：数组索引顺序 + 几何范围 min/max)
// =========================================================================
void SolverDriver::exchange_bc_dq_vol() {
    int rank = mpi_.get_rank();
    int iseq = 0; 

    for (size_t nb = 0; nb < blocks_.size(); ++nb) {
        Block* b_meta = blocks_[nb];
        //if (!b_meta) continue;

        int id_src = b_meta->owner_rank;
        int nregions = b_meta->boundaries.size();

        for (int nr = 0; nr < nregions; ++nr) {
            BoundaryPatch& patch = b_meta->boundaries[nr];
            int ibctype = patch.type;

            if (ibctype < 0) { 
                int nbt = patch.target_block; 
                Block* b_target_meta = blocks_[nbt];
                //if (!b_target_meta) continue;

                int id_des = b_target_meta->owner_rank;

                if (id_src != id_des) {
                    iseq++;
                    int tag_seq = iseq;
                    int nrt = patch.window_id - 1;

                    // ==========================================
                    // 发送方逻辑
                    // ==========================================
                    if (rank == id_src) {
                        Block* b = b_meta;
                        int ng = b->ng;

                        // 1. 确保几何范围正确 (min/max)
                        int ibeg[3], iend[3];
                        for(int k=0; k<3; ++k) {
                            ibeg[k] = std::min(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                            iend[k] = std::max(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                        }

                        // 2. 填充 dqv_t
                        for (int k = ibeg[2]; k <= iend[2]; ++k) {
                            for (int j = ibeg[1]; j <= iend[1]; ++j) {
                                for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                    int ix = (i - 1) + ng;
                                    int iy = (j - 1) + ng;
                                    int iz = (k - 1) + ng;

                                    int bi = i - ibeg[0];
                                    int bj = j - ibeg[1];
                                    int bk = k - ibeg[2];

                                    real_t vol_p = b->vol(ix, iy, iz);
                                    if (vol_p < 1.0e-30) vol_p = 1.0e-30;

                                    for (int m = 0; m < 5; ++m) {
                                        patch.dqv_t(m, bi, bj, bk) = b->dq(ix, iy, iz, m) / vol_p;
                                    }
                                }
                            }
                        }

                        size_t dim_i = iend[0] - ibeg[0] + 1;
                        size_t dim_j = iend[1] - ibeg[1] + 1;
                        size_t dim_k = iend[2] - ibeg[2] + 1;
                        size_t packsize = dim_i * dim_j * dim_k * 5;

                        MPI_Send(patch.dqv_t.host_data(), packsize, MPI_DOUBLE, id_des, tag_seq, MPI_COMM_WORLD);
                    }

                    // ==========================================
                    // 接收方逻辑
                    // ==========================================
                    if (rank == id_des) {
                        Block* b_target = blocks_[nbt];
                        BoundaryPatch& target_patch = b_target->boundaries[nrt];

                        // 使用 Target Patch 的几何范围
                        int ibeg[3], iend[3];
                        for (int k = 0; k < 3; ++k) {
                            ibeg[k] = std::min(std::abs(target_patch.raw_target_is[k]),
                                            std::abs(target_patch.raw_target_ie[k]));
                            iend[k] = std::max(std::abs(target_patch.raw_target_is[k]),
                                            std::abs(target_patch.raw_target_ie[k]));
                        }

                        size_t dim_i = iend[0] - ibeg[0] + 1;
                        size_t dim_j = iend[1] - ibeg[1] + 1;
                        size_t dim_k = iend[2] - ibeg[2] + 1;
                        size_t packsize = dim_i * dim_j * dim_k * 5;

                        MPI_Recv(target_patch.dqvpack_t.host_data(), packsize, MPI_DOUBLE, id_src, tag_seq, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                } 
            } 
        } 
    } 
}

// =========================================================================
// 交换边界 原始变量 (rho, u, v, w, p, t)
// 对应 Fortran: exchange_bc_pv_vol (但在 Fortran 中实际上并不除以体积)
// =========================================================================
void SolverDriver::exchange_bc_pv_vol() {
    int rank = mpi_.get_rank();
    int iseq = 0; 

    for (size_t nb = 0; nb < blocks_.size(); ++nb) {
        Block* b_meta = blocks_[nb];
        //if (!b_meta) continue;

        int id_src = b_meta->owner_rank;
        int nregions = b_meta->boundaries.size();

        for (int nr = 0; nr < nregions; ++nr) {
            BoundaryPatch& patch = b_meta->boundaries[nr];
            int ibctype = patch.type;

            if (ibctype < 0) { 
                int nbt = patch.target_block; 
                Block* b_target_meta = blocks_[nbt];
                //if (!b_target_meta) continue;

                int id_des = b_target_meta->owner_rank;

                if (id_src != id_des) {
                    iseq++;
                    int tag_seq = iseq;
                    int nrt = patch.window_id - 1;

                    // ==========================================
                    // 发送方逻辑
                    // ==========================================
                    if (rank == id_src) {
                        Block* b = b_meta;
                        int ng = b->ng;

                        // 1. 确保几何范围正确 (min/max)
                        int ibeg[3], iend[3];
                        for(int k=0; k<3; ++k) {
                            ibeg[k] = std::min(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                            iend[k] = std::max(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                        }

                        // 2. 填充 qpv_t (6 variables: r, u, v, w, p, t)
                        for (int k = ibeg[2]; k <= iend[2]; ++k) {
                            for (int j = ibeg[1]; j <= iend[1]; ++j) {
                                for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                    int ix = (i - 1) + ng;
                                    int iy = (j - 1) + ng;
                                    int iz = (k - 1) + ng;

                                    int bi = i - ibeg[0];
                                    int bj = j - ibeg[1];
                                    int bk = k - ibeg[2];

                                    // 注意：原始变量直接传递，无需除以体积 (vol_p = 1.0)
                                    // 索引 m=0..5 对应 r, u, v, w, p, t
                                    patch.qpv_t(0, bi, bj, bk) = b->r(ix, iy, iz);
                                    patch.qpv_t(1, bi, bj, bk) = b->u(ix, iy, iz);
                                    patch.qpv_t(2, bi, bj, bk) = b->v(ix, iy, iz);
                                    patch.qpv_t(3, bi, bj, bk) = b->w(ix, iy, iz);
                                    patch.qpv_t(4, bi, bj, bk) = b->p(ix, iy, iz);
                                    patch.qpv_t(5, bi, bj, bk) = b->t(ix, iy, iz);
                                }
                            }
                        }

                        size_t dim_i = iend[0] - ibeg[0] + 1;
                        size_t dim_j = iend[1] - ibeg[1] + 1;
                        size_t dim_k = iend[2] - ibeg[2] + 1;
                        // 包大小: 维度乘积 * 6 (变量数)
                        size_t packsize = dim_i * dim_j * dim_k * 6;

                        MPI_Send(patch.qpv_t.host_data(), packsize, MPI_DOUBLE, id_des, tag_seq, MPI_COMM_WORLD);
                    }

                    // ==========================================
                    // 接收方逻辑
                    // ==========================================
                    if (rank == id_des) {
                        Block* b_target = blocks_[nbt];
                        BoundaryPatch& target_patch = b_target->boundaries[nrt];

                        // 使用 Target Patch 的几何范围
                        int ibeg[3], iend[3];
                        for (int k = 0; k < 3; ++k) {
                            ibeg[k] = std::min(std::abs(target_patch.raw_target_is[k]),
                                            std::abs(target_patch.raw_target_ie[k]));
                            iend[k] = std::max(std::abs(target_patch.raw_target_is[k]),
                                            std::abs(target_patch.raw_target_ie[k]));
                        }

                        size_t dim_i = iend[0] - ibeg[0] + 1;
                        size_t dim_j = iend[1] - ibeg[1] + 1;
                        size_t dim_k = iend[2] - ibeg[2] + 1;
                        size_t packsize = dim_i * dim_j * dim_k * 6;

                        // 接收到 qpvpack_t 中
                        MPI_Recv(target_patch.qpvpack_t.host_data(), packsize, MPI_DOUBLE, id_src, tag_seq, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                } 
            } 
        } 
    } 
}

// =========================================================================
// 对接面平滑 PV (2 Point Match)
// 对应 Fortran: boundary_match_pv_2pm
// =========================================================================
void SolverDriver::boundary_match_pv_2pm() {
    int rank = mpi_.get_rank();

    for (size_t nb = 0; nb < blocks_.size(); ++nb) {
        Block* b = blocks_[nb];
        if (!b || b->owner_rank != rank) continue;

        int nregions = b->boundaries.size();

        for (int nr = 0; nr < nregions; ++nr) {
            BoundaryPatch& patch = b->boundaries[nr];

            if (patch.type < 0) {
                int nbt = patch.target_block; 
                Block* b_target_meta = blocks_[nbt];
                //if (!b_target_meta) continue;

                int id_src = rank;
                int id_des = b_target_meta->owner_rank;

                int ibeg[3], iend[3];
                for (int k = 0; k < 3; ++k) {
                    ibeg[k] = std::min(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                    iend[k] = std::max(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                }
                
                int flat_idx = 0;

                // A. 进程内 (Local)
                if (id_src == id_des) {
                    Block* b_next = blocks_[nbt]; 

                    for (int k = ibeg[2]; k <= iend[2]; ++k) {
                        for (int j = ibeg[1]; j <= iend[1]; ++j) {
                            for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                int ix = (i - 1) + b->ng;
                                int iy = (j - 1) + b->ng;
                                int iz = (k - 1) + b->ng;

                                int it = patch.image[flat_idx];
                                int jt = patch.jmage[flat_idx];
                                int kt = patch.kmage[flat_idx];
                                flat_idx++;

                                int it_x = it + b_next->ng;
                                int jt_y = jt + b_next->ng;
                                int kt_z = kt + b_next->ng;

                                // 原始变量直接平均，无需体积加权
                                b->r(ix, iy, iz) = 0.5 * (b_next->r(it_x, jt_y, kt_z) + b->r(ix, iy, iz));
                                b->u(ix, iy, iz) = 0.5 * (b_next->u(it_x, jt_y, kt_z) + b->u(ix, iy, iz));
                                b->v(ix, iy, iz) = 0.5 * (b_next->v(it_x, jt_y, kt_z) + b->v(ix, iy, iz));
                                b->w(ix, iy, iz) = 0.5 * (b_next->w(it_x, jt_y, kt_z) + b->w(ix, iy, iz));
                                b->p(ix, iy, iz) = 0.5 * (b_next->p(it_x, jt_y, kt_z) + b->p(ix, iy, iz));
                                b->t(ix, iy, iz) = 0.5 * (b_next->t(it_x, jt_y, kt_z) + b->t(ix, iy, iz));
                            }
                        }
                    }
                }
                // B. 跨进程 (MPI)
                else {
                    // 1. 计算 Target 缓冲区的起始偏移量 (0-based from min)
                    int t_st_min[3];
                    t_st_min[0] = std::min(std::abs(patch.raw_target_is[0]), std::abs(patch.raw_target_ie[0]));
                    t_st_min[1] = std::min(std::abs(patch.raw_target_is[1]), std::abs(patch.raw_target_ie[1]));
                    t_st_min[2] = std::min(std::abs(patch.raw_target_is[2]), std::abs(patch.raw_target_ie[2]));

                    for (int k = ibeg[2]; k <= iend[2]; ++k) {
                        for (int j = ibeg[1]; j <= iend[1]; ++j) {
                            for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                int ix = (i - 1) + b->ng;
                                int iy = (j - 1) + b->ng;
                                int iz = (k - 1) + b->ng;

                                int it = patch.image[flat_idx];
                                int jt = patch.jmage[flat_idx];
                                int kt = patch.kmage[flat_idx];
                                flat_idx++;

                                // Global Index (0-based) -> Buffer Index (0-based)
                                // 保持与 boundary_match_dq_2pm 完全一致的索引计算逻辑
                                int buf_i = it - (t_st_min[0] - 1);
                                int buf_j = jt - (t_st_min[1] - 1);
                                int buf_k = kt - (t_st_min[2] - 1);

                                // 从 qpvpack_t 读取邻居值 (6 vars: r, u, v, w, p, t)
                                real_t r_neigh = patch.qpvpack_t(0, buf_i, buf_j, buf_k);
                                real_t u_neigh = patch.qpvpack_t(1, buf_i, buf_j, buf_k);
                                real_t v_neigh = patch.qpvpack_t(2, buf_i, buf_j, buf_k);
                                real_t w_neigh = patch.qpvpack_t(3, buf_i, buf_j, buf_k);
                                real_t p_neigh = patch.qpvpack_t(4, buf_i, buf_j, buf_k);
                                real_t t_neigh = patch.qpvpack_t(5, buf_i, buf_j, buf_k);

                                // 平均更新
                                b->r(ix, iy, iz) = 0.5 * (r_neigh + b->r(ix, iy, iz));
                                b->u(ix, iy, iz) = 0.5 * (u_neigh + b->u(ix, iy, iz));
                                b->v(ix, iy, iz) = 0.5 * (v_neigh + b->v(ix, iy, iz));
                                b->w(ix, iy, iz) = 0.5 * (w_neigh + b->w(ix, iy, iz));
                                b->p(ix, iy, iz) = 0.5 * (p_neigh + b->p(ix, iy, iz));
                                b->t(ix, iy, iz) = 0.5 * (t_neigh + b->t(ix, iy, iz));
                            }
                        }
                    }
                }
            } 
        } 
    } 
}

// =========================================================================
// 两点对接面匹配 (修复：Target 偏移量计算 + 索引顺序确认)
// =========================================================================
void SolverDriver::boundary_match_dq_2pm() {
    int rank = mpi_.get_rank();

    for (size_t nb = 0; nb < blocks_.size(); ++nb) {
        Block* b = blocks_[nb];
        if (!b || b->owner_rank != rank) continue;

        int nregions = b->boundaries.size();

        for (int nr = 0; nr < nregions; ++nr) {
            BoundaryPatch& patch = b->boundaries[nr];

            if (patch.type < 0) {
                int nbt = patch.target_block; 
                Block* b_target_meta = blocks_[nbt];
                //if (!b_target_meta) continue;

                int id_src = rank;
                int id_des = b_target_meta->owner_rank;

                int ibeg[3], iend[3];
                for (int k = 0; k < 3; ++k) {
                    ibeg[k] = std::min(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                    iend[k] = std::max(std::abs(patch.raw_is[k]), std::abs(patch.raw_ie[k]));
                }
                
                int flat_idx = 0;

                // A. 进程内 (Local)
                if (id_src == id_des) {
                    Block* b_next = blocks_[nbt]; 

                    for (int k = ibeg[2]; k <= iend[2]; ++k) {
                        for (int j = ibeg[1]; j <= iend[1]; ++j) {
                            for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                int ix = (i - 1) + b->ng;
                                int iy = (j - 1) + b->ng;
                                int iz = (k - 1) + b->ng;

                                int it = patch.image[flat_idx];
                                int jt = patch.jmage[flat_idx];
                                int kt = patch.kmage[flat_idx];
                                flat_idx++;

                                int it_x = it + b_next->ng;
                                int jt_y = jt + b_next->ng;
                                int kt_z = kt + b_next->ng;

                                real_t vol_p = b->vol(ix, iy, iz) / b_next->vol(it_x, jt_y, kt_z);

                                for (int m = 0; m < 5; ++m) {
                                    real_t dq_neigh = b_next->dq(it_x, jt_y, kt_z, m);
                                    real_t dq_self = b->dq(ix, iy, iz, m);
                                    
                                    b->dq(ix, iy, iz, m) = 0.5 * (dq_neigh * vol_p + dq_self);
                                }
                            }
                        }
                    }
                }
                // B. 跨进程 (MPI)
                else {
                    // 1. 计算 Target 缓冲区的起始偏移量 (0-based from min)
                    int t_st_min[3];
                    t_st_min[0] = std::min(std::abs(patch.raw_target_is[0]), std::abs(patch.raw_target_ie[0]));
                    t_st_min[1] = std::min(std::abs(patch.raw_target_is[1]), std::abs(patch.raw_target_ie[1]));
                    t_st_min[2] = std::min(std::abs(patch.raw_target_is[2]), std::abs(patch.raw_target_ie[2]));

                    for (int k = ibeg[2]; k <= iend[2]; ++k) {
                        for (int j = ibeg[1]; j <= iend[1]; ++j) {
                            for (int i = ibeg[0]; i <= iend[0]; ++i) {
                                int ix = (i - 1) + b->ng;
                                int iy = (j - 1) + b->ng;
                                int iz = (k - 1) + b->ng;

                                int it = patch.image[flat_idx];
                                int jt = patch.jmage[flat_idx];
                                int kt = patch.kmage[flat_idx];
                                flat_idx++;

                                // Global Index (0-based) -> Buffer Index (0-based)
                                // 注意：t_st_min 是 1-based 的 Fortran 坐标，所以要 -1
                                int buf_i = it - (t_st_min[0] - 1);
                                int buf_j = jt - (t_st_min[1] - 1);
                                int buf_k = kt - (t_st_min[2] - 1);

                                real_t vol_p = b->vol(ix, iy, iz);

                                for (int m = 0; m < 5; ++m) {
                                    // 索引顺序 (i, j, k, m)
                                    real_t dq_vol_neigh = patch.dqvpack_t(m, buf_i, buf_j, buf_k);
                                    real_t dq_self = b->dq(ix, iy, iz, m);

                                    // 公式复刻
                                    b->dq(ix, iy, iz, m) = 0.5 * (dq_vol_neigh * vol_p + dq_self);
                                }
                            }
                        }
                    }
                }
            } 
        } 
    } 
}

void SolverDriver::calculate_inviscous_rhs() {
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
// 核心计算：Roe 通量格式 (100% 复刻 src-fortran/flux.f90)
// =========================================================================
void SolverDriver::flux_roe_kernel(
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

// =========================================================================
// 驱动函数：计算一条线上的通量导数 (WCNS + Roe)
// 1. 调用 value_half_node_static 计算几何
// 2. 调用 flux_dxyz_static 计算通量导数
// 3. 严格复刻 Fortran 边界迭代逻辑 (Backup 1..4)
// =========================================================================
void SolverDriver::flux_line_wcns_roe(
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
// 计算无粘右端项 (Inviscid RHS) - 框架实现
// 对应 Fortran: subroutine inviscd3d
// =========================================================================
void SolverDriver::compute_block_inviscous_rhs(Block* b) {
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
// 隐式时间推进主驱动 (LHS)
// 对应 Fortran: subroutine l_h_s_tgh
// =========================================================================
void SolverDriver::time_step_lhs_tgh() {
    int rank = mpi_.get_rank();
    // 遍历本地 Block (对应 Fortran: do pnb=1,pnblocks)
    // C++ 中 blocks_ 容器只存储属于当前 MPI 进程的 Block，无需额外判断
    for (auto* b : blocks_) {
        
        if (b && b->owner_rank == rank) {
            // 1. 边界处理：将非对接边界的 dq 设为 0 (或处理为 -RHS)
            set_boundary_dq_zero(b);

            // 2. 核心求解：点松弛 Gauss-Seidel 迭代
            solve_gauss_seidel_single(b);

            // 3. 边界处理：再次确保边界条件一致性
            set_boundary_dq_zero(b);

            // 4. 变量更新：Q = Q_old + dq * dt
            update_conservatives(b);
        }
    }
}

// =========================================================================
// 子函数占位符 (待实现)
// =========================================================================

// =========================================================================
// 将非对接边界的 Ghost 区域 dq 设为 0 (防止隐式迭代发散)
// ========================================================================= 
void SolverDriver::set_boundary_dq_zero(Block* b) {
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
// 对应 Fortran: subroutine GS_PR_Single
// =========================================================================
void SolverDriver::solve_gauss_seidel_single(Block* b) {
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
void SolverDriver::store_rhs_nb(Block* b, std::vector<real_t>& rhs_nb) {
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
void SolverDriver::matrix_vector_product_std(
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
void SolverDriver::lusgs_l(Block* b, const std::vector<real_t>& rhs_nb, 
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
void SolverDriver::lusgs_u(Block* b, const std::vector<real_t>& rhs_nb, 
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
void SolverDriver::gs_pr_l(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta) {
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
void SolverDriver::gs_pr_u(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta) {
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

void SolverDriver::update_conservatives(Block* b) {
  const auto& global = GlobalData::getInstance();
  const real_t gama   = global.gamma;
  const real_t gamam1 = gama - real_t(1.0);

  // Fortran: p_min=pmin_limit, ...
  const real_t p_min = real_t(1.0e-8);
  const real_t p_max = real_t(1.0e+3);
  const real_t r_min = real_t(1.0e-8);
  const real_t r_max = real_t(1.0e+3);

  const int ni = b->ni;
  const int nj = b->nj;
  const int nk = b->nk;
  const int ng = b->ng;
  const int nm = 5;

  int n_count = 0;

  std::array<real_t,5> qtem;

  for (int k = 0; k < nk; ++k) {
    for (int j = 0; j < nj; ++j) {
      for (int i = 0; i < ni; ++i) {

        const int I = i + ng;
        const int J = j + ng;
        const int K = k + ng;

        // --- qtem = q + dq (trial) ---
        for (int m = 0; m < nm; ++m) {
          qtem[m] = b->q(I,J,K,m) + b->dq(I,J,K,m);
        }

        real_t rm  = qtem[0];
        real_t rm1 = real_t(1.0) / rm;
        real_t um  = qtem[1] * rm1;
        real_t vm  = qtem[2] * rm1;
        real_t wm  = qtem[3] * rm1;
        real_t em  = qtem[4];
        real_t pm  = gamam1 * ( em - real_t(0.5)*rm*(um*um + vm*vm + wm*wm) );

        // Falpha = 1/(1+2*max(0,-0.3+abs((pm-pold)/pold)))
        const real_t pold = b->p(I,J,K); 
        const real_t dp_p = std::abs((pm - pold) / pold);
        const real_t Falpha = real_t(1.0) /
          ( real_t(1.0) + real_t(2.0) * std::max(real_t(0.0), real_t(-0.3) + dp_p) );

        // --- q = q + dq * Falpha (and store back) ---
        for (int m = 0; m < nm; ++m) {
          qtem[m] = b->q(I,J,K,m) + b->dq(I,J,K,m) * Falpha;
          b->q(I,J,K,m) = qtem[m];
        }

        // --- recompute primitive from updated qtem ---
        rm  = qtem[0];
        rm1 = real_t(1.0) / rm;
        um  = qtem[1] * rm1;
        vm  = qtem[2] * rm1;
        wm  = qtem[3] * rm1;
        em  = qtem[4];
        pm  = gamam1 * ( em - real_t(0.5)*rm*(um*um + vm*vm + wm*wm) );

        // --- if pm/rm out of bounds ---
        if (pm <= p_min || rm <= r_min || pm >= p_max || rm >= r_max) {

          // both branches do the same average; only message differs
          n_count++;

          rm = um = vm = wm = pm = real_t(0.0);
          int np = 0;

          for (int kk = -1; kk <= 1; ++kk) {
            for (int jj = -1; jj <= 1; ++jj) {
              for (int ii = -1; ii <= 1; ++ii) {
                if (std::abs(ii) + std::abs(jj) + std::abs(kk) == 1) {
                  np++;

                  // i0=i+ii (no clamp)
                  const int I0 = I + ii;
                  const int J0 = J + jj;
                  const int K0 = K + kk;

                  rm += b->r(I0,J0,K0);
                  um += b->u(I0,J0,K0);
                  vm += b->v(I0,J0,K0);
                  wm += b->w(I0,J0,K0);
                  pm += b->p(I0,J0,K0);
                }
              }
            }
          }

          const real_t rnp = real_t(np);
          rm /= rnp; um /= rnp; vm /= rnp; wm /= rnp; pm /= rnp;

          em = pm/(gama - real_t(1.0)) + real_t(0.5)*rm*(um*um + vm*vm + wm*wm);

          b->q(I,J,K,0) = rm;
          b->q(I,J,K,1) = rm * um;
          b->q(I,J,K,2) = rm * vm;
          b->q(I,J,K,3) = rm * wm;
          b->q(I,J,K,4) = em;
        }

        // --- Fortran: write primitive arrays every cell (always) ---
        b->r(I,J,K) = rm;
        b->u(I,J,K) = um;
        b->v(I,J,K) = vm;
        b->w(I,J,K) = wm;
        b->p(I,J,K) = pm;
      }
    }
  }

  // --- Fortran: per-block stop condition (NOT global SUM) ---
  const int local_abort =
      (n_count > (ni*nj*nk)/64) || (n_count >= 500);

  // Make all ranks abort together if any rank triggers
  int any_abort = 0;
  MPI_Allreduce(&local_abort, &any_abort, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (any_abort) {
    if (mpi_.is_root()) {
      std::cerr << "update_limited(strict): too many unphysical points. "
                << "local n_count=" << n_count
                << " (threshold=" << (ni*nj*nk)/64 << ", 500)\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

// =========================================================================
// 输出模块：VTK XML Structured Grid (.vts) + MultiBlock (.vtm)
// 修正：文件名索引 0-based 统一，增加目录创建，严格归属权判断
// =========================================================================
void SolverDriver::output_solution(int step_idx) {
    int rank = mpi_.get_rank();

    // 1. 准备输出目录
    std::string out_dir = config_->files.output_dir;
    if (out_dir.empty()) out_dir = ".";
    if (out_dir.back() != '/') out_dir += "/";

    // 尝试创建目录 (仅 Rank 0 尝试，出错忽略，假设用户有权限)
    if (rank == 0) {
#ifdef _WIN32
        _mkdir(out_dir.c_str());
#else
        mkdir(out_dir.c_str(), 0755);
#endif
    }
    // 确保目录创建完成
    MPI_Barrier(MPI_COMM_WORLD);

    // 格式化时间步字符串 (例如 001000)
    std::ostringstream ss_step;
    ss_step << std::setw(6) << std::setfill('0') << step_idx;
    std::string step_str = ss_step.str();

    // 2. 并行输出本地 Blocks (.vts)
    int local_written_count = 0;
    
    for (auto* b : blocks_) {
        // 【关键判断 1】Slave 进程中不属于自己的块是 nullptr
        if (b == nullptr) continue;

        // 【关键判断 2】Master 进程中存有所有 Block 的 Shell，但只有 owner_rank 为 0 的才包含数据
        // 必须严格检查 owner_rank 是否等于当前 rank
        if (b->owner_rank != rank) continue;

        // 【关键修正】使用 0-based ID 生成文件名，保持与 VTM 索引一致
        int bid = b->id; 
        
        std::ostringstream ss_blk;
        // 格式: flow_001000_block_0000.vts
        ss_blk << "flow_" << step_str << "_block_" << std::setw(4) << std::setfill('0') << bid << ".vts";
        std::string filename = out_dir + ss_blk.str();

        write_block_vts(b, filename);
        local_written_count++;
    }

    // 3. 全局同步：确保所有 VTS 文件都已写入磁盘
    MPI_Barrier(MPI_COMM_WORLD);

    // 4. Rank 0 生成全局索引 (.vtm)
    if (rank == 0) {
        std::string vtm_filename = out_dir + "flow_" + step_str + ".vtm";
        
        // 获取总块数 (从 GlobalData 获取)
        int total_blocks = GlobalData::getInstance().n_blocks; 
        
        if (total_blocks > 0) {
            write_global_vtm(vtm_filename, step_str, total_blocks);
            spdlog::info(">>> Output: {} (Total Blocks: {})", vtm_filename, total_blocks);
        } else {
            spdlog::error(">>> Error: GlobalData.n_blocks is 0. VTM not written.");
        }
    }
}

// -------------------------------------------------------------------------
// 写单个 Block 的 VTS 文件 (Binary Appended, Interleaved Data)
// -------------------------------------------------------------------------
void SolverDriver::write_block_vts(Block* b, const std::string& filename) {
    std::ofstream out(filename, std::ios::out | std::ios::binary);
    if (!out) {
        spdlog::error("Error: Cannot open file {} for writing.", filename);
        return;
    }

    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;
    size_t n_points = (size_t)ni * nj * nk;
    
    // 数据头类型 (UInt64 = 8 bytes)
    using OffsetType = uint64_t;
    const OffsetType header_sz = sizeof(OffsetType);
    const size_t double_sz = sizeof(double);

    // --- 计算偏移量 (Offsets) ---
    // 顺序必须与 XML 中的 DataArray 写入顺序严格一致
    OffsetType current_offset = 0;

    // 1. Points (3 components: x, y, z)
    OffsetType off_coords = current_offset;
    current_offset += header_sz + n_points * 3 * double_sz;

    // 2. Density (Scalar)
    OffsetType off_rho = current_offset;
    current_offset += header_sz + n_points * 1 * double_sz;

    // 3. Velocity (Vector: u, v, w)
    OffsetType off_vel = current_offset;
    current_offset += header_sz + n_points * 3 * double_sz;

    // 4. Pressure (Scalar)
    OffsetType off_p = current_offset;
    current_offset += header_sz + n_points * 1 * double_sz;

    // 5. Temperature (Scalar)
    OffsetType off_t = current_offset;
    current_offset += header_sz + n_points * 1 * double_sz;

    // 6. Mach (Scalar)
    OffsetType off_ma = current_offset;
    current_offset += header_sz + n_points * 1 * double_sz;

    // --- A. XML Header ---
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    // WholeExtent 是闭区间 [0, ni-1]
    out << "  <StructuredGrid WholeExtent=\"0 " << ni-1 << " 0 " << nj-1 << " 0 " << nk-1 << "\">\n";
    out << "    <Piece Extent=\"0 " << ni-1 << " 0 " << nj-1 << " 0 " << nk-1 << "\">\n";
    
    // Coordinates
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_coords << "\"/>\n";
    out << "      </Points>\n";

    // Point Data
    out << "      <PointData Scalars=\"Density,Pressure,Temperature,Mach\" Vectors=\"Velocity\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"Density\" format=\"appended\" offset=\"" << off_rho << "\"/>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_vel << "\"/>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Pressure\" format=\"appended\" offset=\"" << off_p << "\"/>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Temperature\" format=\"appended\" offset=\"" << off_t << "\"/>\n";
    out << "        <DataArray type=\"Float64\" Name=\"Mach\" format=\"appended\" offset=\"" << off_ma << "\"/>\n";
    out << "      </PointData>\n";
    
    out << "    </Piece>\n";
    out << "  </StructuredGrid>\n";
    
    // --- B. Appended Data Section ---
    out << "  <AppendedData encoding=\"raw\">\n";
    out << "_"; // Start marker

    // Lambda: 写数据块 (Header + Payload)
    auto write_chunk = [&](const std::vector<double>& data) {
        OffsetType size_bytes = data.size() * sizeof(double);
        out.write(reinterpret_cast<const char*>(&size_bytes), sizeof(OffsetType));
        out.write(reinterpret_cast<const char*>(data.data()), size_bytes);
    };

    // --- 1. Write Points (Interleaved X, Y, Z) ---
    {
        std::vector<double> buf; 
        buf.reserve(n_points * 3);
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    // 访问物理区：加上 ghost 偏移 ng
                    int I = i + ng; int J = j + ng; int K = k + ng;
                    buf.push_back(b->x(I, J, K));
                    buf.push_back(b->y(I, J, K));
                    buf.push_back(b->z(I, J, K));
                }
            }
        }
        write_chunk(buf);
    }

    // --- 2. Write Density ---
    {
        std::vector<double> buf; buf.reserve(n_points);
        for (int k = 0; k < nk; ++k)
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i)
                    buf.push_back(b->r(i+ng, j+ng, k+ng));
        write_chunk(buf);
    }

    // --- 3. Write Velocity (Interleaved U, V, W) ---
    {
        std::vector<double> buf; buf.reserve(n_points * 3);
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    int I = i + ng; int J = j + ng; int K = k + ng;
                    buf.push_back(b->u(I, J, K));
                    buf.push_back(b->v(I, J, K));
                    buf.push_back(b->w(I, J, K));
                }
            }
        }
        write_chunk(buf);
    }

    // --- 4. Write Pressure ---
    {
        std::vector<double> buf; buf.reserve(n_points);
        for (int k = 0; k < nk; ++k)
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i)
                    buf.push_back(b->p(i+ng, j+ng, k+ng));
        write_chunk(buf);
    }

    // --- 5. Write Temperature ---
    {
        std::vector<double> buf; buf.reserve(n_points);
        for (int k = 0; k < nk; ++k)
            for (int j = 0; j < nj; ++j)
                for (int i = 0; i < ni; ++i)
                    buf.push_back(b->t(i+ng, j+ng, k+ng));
        write_chunk(buf);
    }

    // --- 6. Write Mach Number ---
    {
        real_t gamma = GlobalData::getInstance().gamma;
        std::vector<double> buf; buf.reserve(n_points);
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    int I = i + ng; int J = j + ng; int K = k + ng;
                    
                    double u_val = b->u(I, J, K);
                    double v_val = b->v(I, J, K);
                    double w_val = b->w(I, J, K);
                    double p_val = b->p(I, J, K);
                    double r_val = b->r(I, J, K);
                    
                    double v2 = u_val*u_val + v_val*v_val + w_val*w_val;
                    // 简单的声速计算与保护
                    double c2 = (r_val > 1.0e-30) ? (gamma * p_val / r_val) : 1.0e-30;
                    if (c2 <= 0.0) c2 = 1.0e-30;

                    double ma = std::sqrt(v2 / c2);
                    buf.push_back(ma);
                }
            }
        }
        write_chunk(buf);
    }

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";
    out.close();
}

// -------------------------------------------------------------------------
// 写全局 VTM 索引文件 (Rank 0)
// -------------------------------------------------------------------------
void SolverDriver::write_global_vtm(const std::string& filename, const std::string& step_str, int total_blocks) {
    std::ofstream out(filename);
    if (!out) {
        spdlog::error("Error: Cannot open VTM file {} for writing.", filename);
        return;
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <vtkMultiBlockDataSet>\n";

    // 关键修正：Block ID 必须从 0 开始计数，直到 total_blocks-1
    // 以匹配 SimulationLoader 中 new Block(i, ...) 的创建逻辑
    for (int id = 0; id < total_blocks; ++id) {
        std::ostringstream ss_blk;
        // 文件名必须与 write_block_vts 中生成的一致 (0-based)
        ss_blk << "flow_" << step_str << "_block_" << std::setw(4) << std::setfill('0') << id << ".vts";
        
        // DataSet index 也是 0-based
        out << "    <DataSet index=\"" << id << "\" file=\"" << ss_blk.str() << "\"/>\n";
    }

    out << "  </vtkMultiBlockDataSet>\n";
    out << "</VTKFile>\n";
    out.close();
}

// =========================================================================
// 输出模块：Binary Tecplot (.plt) Serial Output (Safe Version)
// 修正：移除 std::function，展平循环以避免 Lambda 捕获问题，增强安全性
// =========================================================================
void SolverDriver::output_solution_plt(int step_idx) {
    int rank = mpi_.get_rank();
    auto& global = GlobalData::getInstance();
    real_t gamma = global.gamma;

    // 1. 准备文件名与目录
    std::string out_dir = config_->files.output_dir;
    if (out_dir.empty()) out_dir = ".";
    if (out_dir.back() != '/') out_dir += "/";

    if (rank == 0) {
#ifdef _WIN32
        _mkdir(out_dir.c_str());
#else
        mkdir(out_dir.c_str(), 0755);
#endif
    }
    // 确保所有进程都到达此处
    MPI_Barrier(MPI_COMM_WORLD);

    std::ostringstream ss;
    ss << out_dir << "flow_" << std::setw(6) << std::setfill('0') << step_idx << ".plt";
    std::string filename = ss.str();

    // 2. 收集所有 Block 的元数据 (ID, Size, Owner)
    std::vector<BlockMeta> global_metas = gather_all_block_metas();
    int n_blocks = global_metas.size();

    // 3. Rank 0 打开文件并写入总文件头
    std::ofstream out;
    if (rank == 0) {
        out.open(filename, std::ios::out | std::ios::binary);
        if (!out) {
            spdlog::error("Failed to open PLT file for writing: {}", filename);
            return;
        }
        // 变量名: X,Y,Z,R,U,V,W,P,T,M
        std::vector<std::string> var_names = {"X", "Y", "Z", "R", "U", "V", "W", "P", "T", "M"};
        write_plt_header(out, n_blocks, var_names);
        spdlog::info(">>> Writing PLT: {} (Total Blocks: {})", filename, n_blocks);
    }

    // 4. 逐个 Block 处理 (按全局 ID 顺序)
    for (int bid = 0; bid < n_blocks; ++bid) {
        const auto& meta = global_metas[bid];
        int owner = meta.owner_rank;
        int ni = meta.ni;
        int nj = meta.nj;
        int nk = meta.nk;
        size_t n_points = (size_t)ni * nj * nk;
        int n_vars = 10;
        
        // 跳过空块（如果有）
        if (n_points == 0) continue;

        std::vector<double> buffer; // Rank 0 专用接收/写入缓冲区

        // --- A. 数据收集阶段 ---
        
        // 情况 1: Rank 0 是 Owner (本地直接填充)
        if (rank == 0 && owner == 0) {
            Block* b = nullptr;
            for (auto* lb : blocks_) { 
                if (lb && lb->id == bid) { b = lb; break; } 
            }

            if (b) {
                buffer.resize(n_points * n_vars);
                int ng = b->ng;
                
                // 直接展开循环填充，避免 Lambda 引用问题
                size_t p = 0;
                for(int k=0; k<nk; ++k)
                for(int j=0; j<nj; ++j)
                for(int i=0; i<ni; ++i) {
                    // 使用局部变量缓存物理坐标索引
                    int I = i + ng; int J = j + ng; int K = k + ng;
                    
                    // 按变量块顺序存储: 0:X, 1:Y, 2:Z, 3:R, 4:U, 5:V, 6:W, 7:P, 8:T, 9:M
                    buffer[0 * n_points + p] = b->x(I, J, K);
                    buffer[1 * n_points + p] = b->y(I, J, K);
                    buffer[2 * n_points + p] = b->z(I, J, K);
                    buffer[3 * n_points + p] = b->r(I, J, K);
                    buffer[4 * n_points + p] = b->u(I, J, K);
                    buffer[5 * n_points + p] = b->v(I, J, K);
                    buffer[6 * n_points + p] = b->w(I, J, K);
                    buffer[7 * n_points + p] = b->p(I, J, K);
                    buffer[8 * n_points + p] = b->t(I, J, K);
                    
                    // Mach 计算
                    double r_val = b->r(I, J, K);
                    double p_val = b->p(I, J, K);
                    double u_val = b->u(I, J, K);
                    double v_val = b->v(I, J, K);
                    double w_val = b->w(I, J, K);
                    double vv = u_val*u_val + v_val*v_val + w_val*w_val;
                    double c2 = (r_val > 1.0e-30) ? (gamma * p_val / r_val) : 1.0e-30;
                    if(c2 <= 0.0) c2 = 1.0e-30;
                    buffer[9 * n_points + p] = std::sqrt(vv / c2);
                    
                    p++;
                }
            } else {
                spdlog::error("Rank 0 owns block {} but cannot find it in memory!", bid);
                // 填充零以防止写入错位
                buffer.assign(n_points * n_vars, 0.0);
            }
        }
        // 情况 2: Rank 0 不是 Owner (接收数据)
        else if (rank == 0 && owner != 0) {
            buffer.resize(n_points * n_vars);
            // 阻塞接收，确保数据同步
            MPI_Recv(buffer.data(), n_points * n_vars, MPI_DOUBLE, owner, bid, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 情况 3: Slave 是 Owner (发送数据)
        else if (rank == owner) {
            Block* b = nullptr;
            for (auto* lb : blocks_) { 
                if (lb && lb->id == bid) { b = lb; break; } 
            }

            if (b) {
                std::vector<double> send_buf(n_points * n_vars);
                int ng = b->ng;
                
                size_t p = 0;
                for(int k=0; k<nk; ++k)
                for(int j=0; j<nj; ++j)
                for(int i=0; i<ni; ++i) {
                    int I = i + ng; int J = j + ng; int K = k + ng;
                    
                    send_buf[0 * n_points + p] = b->x(I, J, K);
                    send_buf[1 * n_points + p] = b->y(I, J, K);
                    send_buf[2 * n_points + p] = b->z(I, J, K);
                    send_buf[3 * n_points + p] = b->r(I, J, K);
                    send_buf[4 * n_points + p] = b->u(I, J, K);
                    send_buf[5 * n_points + p] = b->v(I, J, K);
                    send_buf[6 * n_points + p] = b->w(I, J, K);
                    send_buf[7 * n_points + p] = b->p(I, J, K);
                    send_buf[8 * n_points + p] = b->t(I, J, K);
                    
                    double r_val = b->r(I, J, K);
                    double p_val = b->p(I, J, K);
                    double vv = b->u(I, J, K)*b->u(I, J, K) + b->v(I, J, K)*b->v(I, J, K) + b->w(I, J, K)*b->w(I, J, K);
                    double c2 = (r_val > 1.0e-30) ? (gamma * p_val / r_val) : 1.0e-30;
                    if(c2 <= 0.0) c2 = 1.0e-30;
                    send_buf[9 * n_points + p] = std::sqrt(vv / c2);
                    
                    p++;
                }
                
                MPI_Send(send_buf.data(), n_points * n_vars, MPI_DOUBLE, 0, bid, MPI_COMM_WORLD);
            } else {
                spdlog::error("Rank {} owns block {} but cannot find it in memory!", rank, bid);
                // 严重错误：如果找不到块，Rank 0 会死锁在 Recv。必须中止或发送假数据。
                // 这里我们发送全零以防止死锁
                std::vector<double> dummy(n_points * n_vars, 0.0);
                MPI_Send(dummy.data(), n_points * n_vars, MPI_DOUBLE, 0, bid, MPI_COMM_WORLD);
            }
        }

        // --- B. 写入阶段 (仅 Rank 0) ---
        if (rank == 0 && out.is_open()) {
            write_plt_zone_header(out, bid, ni, nj, nk);
            write_plt_data_block(out, buffer);
        }
    }

    if (rank == 0 && out.is_open()) {
        out.close();
    }
    
    // 确保所有通信完成
    MPI_Barrier(MPI_COMM_WORLD);
}
// -------------------------------------------------------------------------
// 辅助函数实现
// -------------------------------------------------------------------------

std::vector<SolverDriver::BlockMeta> SolverDriver::gather_all_block_metas() {
    int rank = mpi_.get_rank();
    int size = mpi_.get_size();
    
    // 1. 本地统计
    int n_blocks_global = GlobalData::getInstance().n_blocks;
    std::vector<BlockMeta> local_metas(n_blocks_global);
    
    // 初始化为无效
    for(auto& m : local_metas) { m.id = -1; m.owner_rank = -1; }

    for(auto* b : blocks_) {
        if(b && b->owner_rank == rank) {
            local_metas[b->id].id = b->id;
            local_metas[b->id].ni = b->ni;
            local_metas[b->id].nj = b->nj;
            local_metas[b->id].nk = b->nk;
            local_metas[b->id].owner_rank = rank;
        }
    }

    // 2. 归约到 Rank 0
    // 定义 MPI Struct 比较麻烦，这里我们拆分数组进行 Reduce
    // 实际上 Rank 0 需要知道每个 Block ID 对应的 owner 和 size
    
    // 简单的做法：因为 Block ID 是 0~N-1 连续的，我们可以用 4 个数组
    std::vector<int> send_owners(n_blocks_global), recv_owners(n_blocks_global);
    std::vector<int> send_ni(n_blocks_global), recv_ni(n_blocks_global);
    std::vector<int> send_nj(n_blocks_global), recv_nj(n_blocks_global);
    std::vector<int> send_nk(n_blocks_global), recv_nk(n_blocks_global);

    for(int i=0; i<n_blocks_global; ++i) {
        send_owners[i] = local_metas[i].owner_rank;
        send_ni[i] = (local_metas[i].id != -1) ? local_metas[i].ni : 0;
        send_nj[i] = (local_metas[i].id != -1) ? local_metas[i].nj : 0;
        send_nk[i] = (local_metas[i].id != -1) ? local_metas[i].nk : 0;
    }

    // 使用 MPI_MAX 归约 (owner_rank 初始化为 -1, size 初始化为 0)
    MPI_Reduce(send_owners.data(), recv_owners.data(), n_blocks_global, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(send_ni.data(), recv_ni.data(), n_blocks_global, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(send_nj.data(), recv_nj.data(), n_blocks_global, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(send_nk.data(), recv_nk.data(), n_blocks_global, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // 广播回所有进程 (因为 Slave 发送时也需要知道 owner 是否是自己 - 虽然本地已有，但保持一致性更好)
    // 实际上 output_solution_plt 逻辑中，Rank 0 驱动循环，Slave 只需要判断 "我是不是 owner"。
    // Slave 已经知道自己拥有哪些块。
    // 但是 Rank 0 需要知道 "谁拥有 Block i"。所以 recv 数组只在 Rank 0 有效即可。
    
    std::vector<BlockMeta> result;
    if (rank == 0) {
        result.resize(n_blocks_global);
        for(int i=0; i<n_blocks_global; ++i) {
            result[i].id = i;
            result[i].owner_rank = recv_owners[i];
            result[i].ni = recv_ni[i];
            result[i].nj = recv_nj[i];
            result[i].nk = recv_nk[i];
        }
    }
    
    // 为了让 Slave 也能进入 bid 循环 (虽然它不需要知道其他人的信息)，
    // 我们需要把 block 总数广播一下，或者 Slave 直接用 GlobalData.n_blocks。
    // 为了 output_solution_plt 中逻辑简单，我们把完整 meta 广播给所有人。
    
    // 序列化广播
    // 这里简单处理：直接广播 4 个 vector
    MPI_Bcast(recv_owners.data(), n_blocks_global, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(recv_ni.data(), n_blocks_global, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(recv_nj.data(), n_blocks_global, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(recv_nk.data(), n_blocks_global, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        result.resize(n_blocks_global);
        for(int i=0; i<n_blocks_global; ++i) {
            result[i].id = i;
            result[i].owner_rank = recv_owners[i];
            result[i].ni = recv_ni[i];
            result[i].nj = recv_nj[i];
            result[i].nk = recv_nk[i];
        }
    }

    return result;
}

// 写入 Tecplot 二进制头 (Magic + Title + Variables)
void SolverDriver::write_plt_header(std::ofstream& out, int n_blocks, const std::vector<std::string>& var_names) {
    // Magic Number: "#!TDV112"
    out.write("#!TDV112", 8);

    int i_val;
    float f_val;
    
    // Byte Order: 1 = Little Endian
    i_val = 1; out.write((char*)&i_val, 4);
    
    // File Type: 0 = Full
    i_val = 0; out.write((char*)&i_val, 4);
    
    // Title (String)
    std::string title = "WCNS Flow Solution";
    i_val = 0; // Title 结尾
    while(title.length() > 0 && title.back() == 0) title.pop_back();
    for(char c : title) { int c_int = (int)c; out.write((char*)&c_int, 4); }
    out.write((char*)&i_val, 4); // Null terminator

    // Num Vars
    i_val = var_names.size(); out.write((char*)&i_val, 4);

    // Var Names
    for (const auto& name : var_names) {
        for(char c : name) { int c_int = (int)c; out.write((char*)&c_int, 4); }
        int null_term = 0; out.write((char*)&null_term, 4);
    }
}

// 写入 Zone 头
void SolverDriver::write_plt_zone_header(std::ofstream& out, int zone_id, int ni, int nj, int nk) {
    // Zone Marker
    float marker = 299.0f;
    out.write((char*)&marker, 4);

    // Zone Name
    std::string zname = "BLOCK_" + std::to_string(zone_id);
    for(char c : zname) { int c_int = (int)c; out.write((char*)&c_int, 4); }
    int null_term = 0; out.write((char*)&null_term, 4);

    // Parent Zone, Strand ID, Sol Time, Zone Color (All -1 or 0)
    int i_val = -1; 
    out.write((char*)&i_val, 4); // Parent
    out.write((char*)&i_val, 4); // Strand
    double d_val = 0.0;
    out.write((char*)&d_val, 8); // SolTime
    i_val = -1; out.write((char*)&i_val, 4); // Color
    
    // Zone Type: 0 = Ordered
    i_val = 0; out.write((char*)&i_val, 4);
    
    // Var Mode (0 = All Float/Double)
    i_val = 0; out.write((char*)&i_val, 4); // Spec
    i_val = 0; out.write((char*)&i_val, 4); // Comp

    // Dimensions
    out.write((char*)&ni, 4);
    out.write((char*)&nj, 4);
    out.write((char*)&nk, 4);

    // Aux Data? No. Marker?
    // Ordered Zone ends here. Next is data.
    // 实际上 Tecplot 格式中这里可能还有 Aux Data Marker (357.0f) 或 EOH Marker (357.0f)
    // 简单的 Preplot 格式通常以 357.0f 结束头部
    marker = 357.0f; // EOH
    out.write((char*)&marker, 4);
    
    // Zone Header 结束
}

// 写入数据块 (Raw Double)
void SolverDriver::write_plt_data_block(std::ofstream& out, const std::vector<double>& data) {
    // Tecplot Binary Data: 直接写入 float/double 数组
    // 假设是 Block Format (所有 X, 所有 Y...)
    // 我们 buffer 已经是这个顺序了
    out.write((char*)data.data(), data.size() * sizeof(double));
}
// =========================================================================
// 气动力计算与输出 (Pressure Force Integration)
// 修正点：法向量归一化、Config引用、辅助函数实现
// =========================================================================
void SolverDriver::calculate_aerodynamic_forces(int step_idx) {
    const auto& global = GlobalData::getInstance();
    const auto& force_cfg = config_->forceRef; // 使用 Config 中的参考量
    
    int rank = mpi_.get_rank();

    // 1. 准备统计变量 (Fx, Fy, Fz, Mx, My, Mz)
    double local_F[6] = {0.0}; 
    double global_F[6] = {0.0};

    // 2. 获取参考值
    double poo = global.p_inf; 
    double xref = force_cfg.point_ref[0];
    double yref = force_cfg.point_ref[1];
    double zref = force_cfg.point_ref[2];

    // 3. 遍历本地 Block 进行积分
    for (auto* b : blocks_) {
        // Slave 进程空指针检查 & 归属权检查
        if (!b || b->owner_rank != rank) continue;

        int ng = b->ng;

        // 遍历边界 Patch
        for (const auto& bc : b->boundaries) {
            // 只处理壁面 (根据 Fortran: type 2, 20, 21)
            if (!is_wall_boundary(bc.type)) continue;

            // 获取索引范围 (Fortran 1-based -> C++ 0-based + Ghost)
            int is = std::abs(bc.raw_is[0]);
            int ie = std::abs(bc.raw_ie[0]);
            int js = std::abs(bc.raw_is[1]);
            int je = std::abs(bc.raw_ie[1]);
            int ks = std::abs(bc.raw_is[2]);
            int ke = std::abs(bc.raw_ie[2]);

            // 转为 C++ 物理区索引 (0-based)
            int i_start = std::min(is, ie) - 1;
            int i_end   = std::max(is, ie) - 1;
            int j_start = std::min(js, je) - 1;
            int j_end   = std::max(js, je) - 1;
            int k_start = std::min(ks, ke) - 1;
            int k_end   = std::max(ks, ke) - 1;

            // 判断面方向 (根据 s_nd: 1=I, 2=J, 3=K)
            // 注意：Block.h 中 s_nd 是 int，这里假设遵循 Fortran 约定
            bool i_face = (bc.s_nd == 1);
            bool j_face = (bc.s_nd == 2);
            bool k_face = (bc.s_nd == 3);

            // 确定方向符号 (使用 s_lr: -1 或 1)
            double s_lr = (double)bc.s_lr;

            // 确定循环边界 (面元积分需要遍历 n-1 个单元)
            int i_loop_end = i_face ? i_start : i_end - 1;
            int j_loop_end = j_face ? j_start : j_end - 1;
            int k_loop_end = k_face ? k_start : k_end - 1;

            for (int k = k_start; k <= k_loop_end; ++k) {
                for (int j = j_start; j <= j_loop_end; ++j) {
                    for (int i = i_start; i <= i_loop_end; ++i) {
                        
                        // 获取四个角点的索引
                        int idx_i[4], idx_j[4], idx_k[4];
                        
                        if (i_face) {
                            idx_i[0]=i; idx_j[0]=j;   idx_k[0]=k;
                            idx_i[1]=i; idx_j[1]=j+1; idx_k[1]=k;
                            idx_i[2]=i; idx_j[2]=j+1; idx_k[2]=k+1;
                            idx_i[3]=i; idx_j[3]=j;   idx_k[3]=k+1;
                        } else if (j_face) {
                            idx_i[0]=i;   idx_j[0]=j; idx_k[0]=k;
                            idx_i[1]=i+1; idx_j[1]=j; idx_k[1]=k;
                            idx_i[2]=i+1; idx_j[2]=j; idx_k[2]=k+1;
                            idx_i[3]=i;   idx_j[3]=j; idx_k[3]=k+1;
                        } else { // k_face
                            idx_i[0]=i;   idx_j[0]=j;   idx_k[0]=k;
                            idx_i[1]=i+1; idx_j[1]=j;   idx_k[1]=k;
                            idx_i[2]=i+1; idx_j[2]=j+1; idx_k[2]=k;
                            idx_i[3]=i;   idx_j[3]=j+1; idx_k[3]=k;
                        }

                        // 收集数据
                        double x[4], y[4], z[4], px[4], py[4], pz[4];

                        for (int n = 0; n < 4; ++n) {
                            int I = idx_i[n] + ng; 
                            int J = idx_j[n] + ng;
                            int K = idx_k[n] + ng;

                            x[n] = b->x(I, J, K);
                            y[n] = b->y(I, J, K);
                            z[n] = b->z(I, J, K);

                            // 获取该点的度量 (Metric) - 这是面积矢量
                            double mx=0, my=0, mz=0;
                            if (i_face) {
                                mx = b->kcx(I, J, K); my = b->kcy(I, J, K); mz = b->kcz(I, J, K);
                            } else if (j_face) {
                                mx = b->etx(I, J, K); my = b->ety(I, J, K); mz = b->etz(I, J, K);
                            } else {
                                mx = b->ctx(I, J, K); my = b->cty(I, J, K); mz = b->ctz(I, J, K);
                            }

                            // 【核心修正】归一化法向量 (Unit Normal)
                            // 必须除以模长，否则积分时会变成 面积*面积
                            double area_mag = std::sqrt(mx*mx + my*my + mz*mz);
                            double nx = 0.0, ny = 0.0, nz = 0.0;
                            if (area_mag > 1.0e-30) {
                                nx = mx / area_mag;
                                ny = my / area_mag;
                                nz = mz / area_mag;
                            }

                            // 计算压力差向量 (pn * vec)
                            // Fortran force_dif: pn = 2.0 * (p - poo)
                            double pn = 2.0 * (b->p(I, J, K) - poo);
                            
                            // 计算单元压力向量 (Pressure Vector)
                            // s_lr * nx 指向物体内部，P 指向物体内部，所以正号代表压力阻力
                            px[n] = s_lr * nx * pn;
                            py[n] = s_lr * ny * pn;
                            pz[n] = s_lr * nz * pn;
                        }

                        // 调用积分辅助函数 (计算面积并加权)
                        accumulate_face_force(
                            px[0], py[0], pz[0], px[1], py[1], pz[1], 
                            px[2], py[2], pz[2], px[3], py[3], pz[3],
                            x[0], y[0], z[0], x[1], y[1], z[1],
                            x[2], y[2], z[2], x[3], y[3], z[3],
                            xref, yref, zref,
                            local_F
                        );
                    }
                }
            }
        }
    }

    // 4. MPI 全局求和
    MPI_Allreduce(local_F, global_F, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 5. 系数计算与输出 (Master)
    if (rank == 0) {
        double sref = force_cfg.area_ref;
        double lfref = force_cfg.len_ref_Re;
        
        // 防止除零
        if (sref <= 1.0e-30) sref = 1.0;
        if (lfref <= 1.0e-30) lfref = 1.0;

        double scf = 1.0 / sref;
        double vcm = 1.0 / (sref * lfref);
        
        // Fortran: (2 - nwholefield)
        // 假设 Config 中 is_full_field: 1=Full, 0=Half
        double sym_factor = (2.0 - (double)force_cfg.is_full_field);

        // 计算系数
        double cfx = sym_factor * global_F[0] * scf;
        double cfy = sym_factor * global_F[1] * scf;
        double cfz = sym_factor * global_F[2] * scf;
        double cmx = sym_factor * global_F[3] * vcm;
        double cmy = sym_factor * global_F[4] * vcm;
        double cmz = sym_factor * global_F[5] * vcm;

        // 计算升阻力 (CL, CD)
        double alpha = global.alpha_rad;
        double sina = std::sin(alpha);
        double cosa = std::cos(alpha);
        
        double cl = cfy * cosa - cfx * sina;
        double cd = cfx * cosa + cfy * sina; 

        // 写入文件
        std::ofstream out("aerodynamics.dat", std::ios::app);
        if (out) {
            out << std::setw(8) << step_idx << " "
                << std::scientific << std::setprecision(5)
                << cfx << " " << cfy << " " << cfz << " "
                << cmx << " " << cmy << " " << cmz << " "
                << cd << " " << cl << "\n";
        }
        
        // 屏幕日志
        spdlog::info(">>> Aero: CL = {:.5f}, CD = {:.5f}, CFx = {:.5f}", cl, cd, cfx);
    }
}

// =========================================================================
// 辅助：判断是否为壁面
// =========================================================================
bool SolverDriver::is_wall_boundary(int type) {
    // 对应 Fortran: 2=Solid, 20=Isothermal, 21=Adiabatic
    return (type == 2 || type == 20 || type == 21);
}

// =========================================================================
// 辅助：计算四边形面元积分 (复刻 Fortran getforce)
// =========================================================================
void SolverDriver::accumulate_face_force(
    double px1, double py1, double pz1, double px2, double py2, double pz2,
    double px3, double py3, double pz3, double px4, double py4, double pz4,
    double x1, double y1, double z1, double x2, double y2, double z2,
    double x3, double y3, double z3, double x4, double y4, double z4,
    double xref, double yref, double zref,
    double* F // in/out accumulators
) {
    // 1. 中心点
    double px5 = 0.25 * (px1 + px2 + px3 + px4);
    double py5 = 0.25 * (py1 + py2 + py3 + py4);
    double pz5 = 0.25 * (pz1 + pz2 + pz3 + pz4);

    double x5 = 0.25 * (x1 + x2 + x3 + x4);
    double y5 = 0.25 * (y1 + y2 + y3 + y4);
    double z5 = 0.25 * (z1 + z2 + z3 + z4);

    // 2. 边长 (Length)
    auto dist = [](double ax, double ay, double az, double bx, double by, double bz) {
        return std::sqrt(std::pow(ax-bx,2) + std::pow(ay-by,2) + std::pow(az-bz,2));
    };
    double r12 = dist(x1,y1,z1, x2,y2,z2);
    double r23 = dist(x2,y2,z2, x3,y3,z3);
    double r34 = dist(x3,y3,z3, x4,y4,z4);
    double r41 = dist(x4,y4,z4, x1,y1,z1);
    double r15 = dist(x1,y1,z1, x5,y5,z5);
    double r25 = dist(x2,y2,z2, x5,y5,z5);
    double r35 = dist(x3,y3,z3, x5,y5,z5);
    double r45 = dist(x4,y4,z4, x5,y5,z5);

    // 3. 面积 (Heron formula)
    auto tri_area = [](double a, double b, double c) {
        double p = 0.5 * (a + b + c);
        return std::sqrt(std::max(0.0, p * (p-a) * (p-b) * (p-c)));
    };
    double s1 = tri_area(r12, r15, r25); // Triangle 1-2-5
    double s2 = tri_area(r23, r25, r35); // Triangle 2-3-5
    double s3 = tri_area(r34, r35, r45); // Triangle 3-4-5
    double s4 = tri_area(r41, r45, r15); // Triangle 4-1-5

    // 4. 积分 (对 4 个三角形分别求和)
    auto add_tri = [&](double pxa, double pya, double pza,
                       double pxb, double pyb, double pzb,
                       double xa, double ya, double za,
                       double xb, double yb, double zb,
                       double s) 
    {
        // 压力向量平均
        double pxc = (pxa + pxb + px5) / 3.0;
        double pyc = (pya + pyb + py5) / 3.0;
        double pzc = (pza + pzb + pz5) / 3.0;
        
        // 几何中心平均
        double xc = (xa + xb + x5) / 3.0;
        double yc = (ya + yb + y5) / 3.0;
        double zc = (za + zb + z5) / 3.0;

        // 力 = 平均压力矢量 * 面积 (Force = P_vec * Area)
        double dfx = pxc * s;
        double dfy = pyc * s;
        double dfz = pzc * s;

        // 力矩: r x F
        double dx = xc - xref;
        double dy = yc - yref;
        double dz = zc - zref;

        F[0] += dfx;
        F[1] += dfy;
        F[2] += dfz;
        F[3] += (dy * dfz - dz * dfy); // Mx
        F[4] += (dz * dfx - dx * dfz); // My
        F[5] += (dx * dfy - dy * dfx); // Mz
    };

    add_tri(px1,py1,pz1, px2,py2,pz2, x1,y1,z1, x2,y2,z2, s1);
    add_tri(px2,py2,pz2, px3,py3,pz3, x2,y2,z2, x3,y3,z3, s2);
    add_tri(px3,py3,pz3, px4,py4,pz4, x3,y3,z3, x4,y4,z4, s3);
    add_tri(px4,py4,pz4, px1,py1,pz1, x4,y4,z4, x1,y1,z1, s4);
}
} // namespace Hyres