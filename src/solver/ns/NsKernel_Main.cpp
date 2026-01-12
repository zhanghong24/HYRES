#include "NSKernel.h"
#include <fstream>

namespace Hyres {

NsKernel::NsKernel(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi) 
    : blocks_(blocks), config_(config), mpi_(mpi), boundary_manager_(config, blocks) {}

NsKernel::~NsKernel() = default;

void NsKernel::check_residual(int step) {
    int rank = mpi_.get_rank();
    const int nl = 5;
    
    // 配置参数
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
                     step, rms_residual, global_max_res.val, loc_str);
    }
}

// =========================================================================
// 核心函数：计算气动力系数 (纯文件输出版)
// =========================================================================
void NsKernel::calculate_aerodynamic_forces(int step) {
    
    AeroCoeffs local_sum;
    
    // 1. 获取参考压力
    real_t p_ref = config_->inflow.P_ref; 
    if (p_ref <= 0.0) p_ref = 1.0 / GlobalData::getInstance().gamma;

    // 2. 本地积分
    for (auto* b : blocks_) {
        if (b && b->owner_rank == mpi_.get_rank()) {
            integrate_wall_forces(b, local_sum, p_ref);
        }
    }

    // 3. MPI 规约
    std::vector<double> send_buf = {
        (double)local_sum.Fx, (double)local_sum.Fy, (double)local_sum.Fz,
        (double)local_sum.Mx, (double)local_sum.My, (double)local_sum.Mz
    };
    std::vector<double> recv_buf(6, 0.0);

    MPI_Reduce(send_buf.data(), recv_buf.data(), 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // 4. Rank 0 输出到文件
    if (mpi_.get_rank() == 0) {
        real_t cfx = recv_buf[0];
        real_t cfy = recv_buf[1];
        real_t cfz = recv_buf[2];
        real_t cmx = recv_buf[3];
        real_t cmy = recv_buf[4];
        real_t cmz = recv_buf[5];

        // --- 系数归一化 ---
        real_t sref = config_->forceRef.area_ref;
        real_t lref = config_->forceRef.len_ref_grid;
        // 半模/全模系数
        real_t sym_factor = 2.0 - (real_t)config_->forceRef.is_full_field;

        real_t scf = sym_factor / sref;
        real_t vcm = sym_factor / (sref * lref);

        cfx *= scf; cfy *= scf; cfz *= scf;
        cmx *= vcm; cmy *= vcm; cmz *= vcm;

        // --- 坐标轴转换 ---
        real_t alpha_rad = config_->inflow.alpha * M_PI / 180.0;
        real_t beta_rad  = config_->inflow.sideslip * M_PI / 180.0; 

        real_t sina = std::sin(alpha_rad);
        real_t cosa = std::cos(alpha_rad);
        real_t sinb = std::sin(beta_rad);
        real_t cosb = std::cos(beta_rad);

        // CL, CD (假设 Y 轴为升力方向)
        real_t cl = cfy * cosa - cfx * sina;
        real_t cd = cosb * (cfy * sina + cfx * cosa) + sinb * cfz;

        // 压心 Xcp
        real_t small = 1.0e-30;
        real_t xcp = ((cfy >= 0) ? 1.0 : -1.0) * cmz / (std::abs(cfy) + small);

        // --- 文件输出核心逻辑 ---
        std::string filename = "history_force.dat";
        std::ofstream out;

        // 模式选择：如果是第0步，覆盖重写；否则追加
        // 注意：如果是 Restart 模式，这里可能需要改成始终 Append，根据您的 nstart 判断
        // 这里暂时用 step==0 判断
        if (step == 0) {
            out.open(filename, std::ios::out); // 覆盖模式
            // 写 Tecplot 风格表头
            out << "VARIABLES = \"Step\", \"Cfx\", \"Cfy\", \"Cfz\", \"Cmx\", \"Cmy\", \"Cmz\", \"CD\", \"CL\", \"Xcp\"\n";
        } else {
            out.open(filename, std::ios::app); // 追加模式
        }
        
        if (out.is_open()) {
            out << std::scientific << std::setprecision(6);
            out << std::setw(8)  << step 
                << std::setw(16) << cfx
                << std::setw(16) << cfy
                << std::setw(16) << cfz
                << std::setw(16) << cmx
                << std::setw(16) << cmy
                << std::setw(16) << cmz
                << std::setw(16) << cd
                << std::setw(16) << cl
                << std::setw(16) << xcp
                << "\n";
            out.close();
        }
    }
}

// =========================================================================
// 辅助函数：壁面积分 (完全修正版)
// =========================================================================
void NsKernel::integrate_wall_forces(Block* b, AeroCoeffs& sum, real_t p_ref) {
    int ng = b->ng;
    
    // --- 修正 Config 引用 ---
    real_t xref = config_->forceRef.point_ref[0];
    real_t yref = config_->forceRef.point_ref[1];
    real_t zref = config_->forceRef.point_ref[2];
    
    int nvis = config_->physics.viscous_mode; 
    
    // 雷诺数引用
    real_t re_inv = 1.0;
    if (config_->inflow.reynolds > 1.0) {
        re_inv = 1.0 / config_->inflow.reynolds;
    }

    for (const auto& patch : b->boundaries) {
        // 【核心修正】直接使用整数 2 判断壁面 (匹配您的 NsKernel_Boundary.cpp)
        if (patch.type != 2) continue;

        int dim = patch.s_nd; 
        int dir = patch.s_lr; 
        
        // 获取循环范围
        int is = patch.raw_is[0]; int ie = patch.raw_ie[0];
        int js = patch.raw_is[1]; int je = patch.raw_ie[1];
        int ks = patch.raw_is[2]; int ke = patch.raw_ie[2];

        int i_start = std::min(is, ie); int i_end = std::max(is, ie);
        int j_start = std::min(js, je); int j_end = std::max(js, je);
        int k_start = std::min(ks, ke); int k_end = std::max(ks, ke);

        for (int k = k_start; k <= k_end; ++k) {
            for (int j = j_start; j <= j_end; ++j) {
                for (int i = i_start; i <= i_end; ++i) {
                    
                    int I = i - 1 + ng;
                    int J = j - 1 + ng;
                    int K = k - 1 + ng;

                    // 获取 Metrics (面法向面积矢量)
                    real_t Sx, Sy, Sz;
                    if (dim == 0) {
                        Sx = b->kcx(I, J, K); Sy = b->kcy(I, J, K); Sz = b->kcz(I, J, K);
                    } else if (dim == 1) {
                        Sx = b->etx(I, J, K); Sy = b->ety(I, J, K); Sz = b->etz(I, J, K);
                    } else {
                        Sx = b->ctx(I, J, K); Sy = b->cty(I, J, K); Sz = b->ctz(I, J, K);
                    }

                    // 修正方向: 确保法向指向流体内部还是外部？
                    // Fortran: pnx = s_lr * nx * pn. 
                    // 我们直接乘上 dir (s_lr)
                    real_t dir_f = (real_t)dir; 
                    Sx *= dir_f; Sy *= dir_f; Sz *= dir_f;

                    // 压力系数: 2.0 * (p - p_inf)
                    real_t p_wall = b->p(I, J, K);
                    real_t pn = 2.0 * (p_wall - p_ref);

                    real_t dfx = pn * Sx;
                    real_t dfy = pn * Sy;
                    real_t dfz = pn * Sz;

                    // 粘性部分 (仅当是粘性计算时)
                    // 您在 NsKernel_Boundary.cpp 里区分了 inviscid/viscid wall
                    // 这里我们根据全局物理配置 nvis 判断即可
                    if (nvis > 0) {
                        // [占位] 如果有梯度变量，在此处添加计算
                        // 需要计算 tau_xx 等并投影到 (Sx, Sy, Sz)
                    }

                    sum.Fx += dfx;
                    sum.Fy += dfy;
                    sum.Fz += dfz;

                    real_t x = b->x(I, J, K) - xref;
                    real_t y = b->y(I, J, K) - yref;
                    real_t z = b->z(I, J, K) - zref;

                    sum.Mx += (y * dfz - z * dfy);
                    sum.My += (z * dfx - x * dfz);
                    sum.Mz += (x * dfy - y * dfx);
                }
            }
        }
    }
}

void NsKernel::apply_boundary() {
    
    exchange_bc();
    
    int rank = mpi_.get_rank();
    for (Block* b : blocks_) {
        if (b && b->owner_rank == rank) {
            boundary_manager_.boundary_sequence(b);
        }
    }
}

void NsKernel::compute_time_step() {
    
    update_derived_variables();

    auto& global = GlobalData::getInstance();
    int ntmst = config_->timeStep.mode; 
    int rank = mpi_.get_rank();

    // -------------------------------------
    // 逐块计算 (Loop 1)
    // -------------------------------------
    for (Block* b : blocks_) {
        if (b && b->owner_rank == rank) {
            spectrum_tgh(b);

            localdt0(b);
        }
    }
}

void NsKernel::compute_rhs() {
    // step 1. 计算粘性残差
    if (config_->physics.viscous_mode == 1) {
        calculate_viscous_rhs();
        
        communicate_dq_npp();
    }

    // step 2. 计算无粘残差
    calculate_inviscous_rhs();

    communicate_dq_npp();
}

void NsKernel::compute_lhs() {
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

    communicate_pv_npp();
}

void NsKernel::update_conservatives(Block* b) {
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
void NsKernel::output_solution(int step_idx) {
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

// =========================================================================
// Restart 输出函数：保存原始量 r, u, v, w, p, t
// 文件名格式: restart_flow_{step}.dat
// =========================================================================
void NsKernel::output_restart(int step) {
    int rank = mpi_.get_rank();
    int nblocks = blocks_.size();

    // 1. 生成文件名
    std::string filename = fmt::format("restart_flow.dat");

    std::ofstream out;
    if (rank == 0) {
        out.open(filename, std::ios::binary | std::ios::out);
        if (!out.is_open()) {
            spdlog::error("Failed to open restart file: {}", filename);
            return;
        }
        
        // 2. 写入文件头 (Header)
        // 包含: 步数, 块数, 物理时间 (可选), 甚至可以存个 Magic Number
        out.write(reinterpret_cast<const char*>(&step), sizeof(int));
        out.write(reinterpret_cast<const char*>(&nblocks), sizeof(int));
        
        // 如果有物理时间，也可以写进去，方便非定常续算
        // double time = ...;
        // out.write((char*)&time, sizeof(double));
        
        spdlog::info(">>> Writing Restart File: {}", filename);
    }

    // 3. 逐块写入 (按 ID 顺序)
    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b = blocks_[nb];
        
        // 获取该块的基本信息 (所有进程都知道这些 Meta 信息)
        int ni = b->ni;
        int nj = b->nj;
        int nk = b->nk;
        int ng = b->ng;
        int owner = b->owner_rank;

        size_t num_cells = (size_t)ni * nj * nk;
        int n_vars = 6; // r, u, v, w, p, t
        
        // 准备发送/接收缓冲区 (仅物理网格大小)
        // 结构: [r...][u...][v...][w...][p...][t...] 
        // 也可以是 Point-wise AOQ: [r,u,v,w,p,t]... 这里为了压缩可能选择分开存，
        // 但为了简单和读写一致，建议按变量块存储 (Plot3D 风格) 或者 纯平铺。
        // 这里采用: Point-wise (SOA or AOS). 
        // 为了方便 SimulationLoader::read_plot3d_grid 的习惯，我们采用 SOA (Variable-Fastest 这种说法有歧义)，
        // 即: 先存所有的 r, 再存所有的 u ... 这样比较符合 Fortran 数组习惯。
        
        std::vector<double> buffer;
        
        // --- 数据打包 (Packing) ---
        // 只有 Owner 需要打包数据
        if (rank == owner) {
            buffer.resize(num_cells * n_vars);
            size_t idx = 0;
            
            // 辅助 Lambda: 提取单个变量的纯物理区域
            auto pack_var = [&](const HyresArray<double>& var) {
                for (int k = 0; k < nk; ++k) {
                    for (int j = 0; j < nj; ++j) {
                        for (int i = 0; i < ni; ++i) {
                            // 注意: 加上 ng 偏移
                            buffer[idx++] = var(i + ng, j + ng, k + ng);
                        }
                    }
                }
            };

            // 依次打包 6 个变量
            pack_var(b->r);
            pack_var(b->u);
            pack_var(b->v);
            pack_var(b->w);
            pack_var(b->p);
            pack_var(b->t);
        }

        // --- 数据传输与写入 ---
        
        if (rank == 0) {
            if (owner == 0) {
                // 本地块：直接写
                out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(double));
            } else {
                // 远程块：接收后写
                // 需要重新分配 buffer 大小以接收数据
                std::vector<double> recv_buf(num_cells * n_vars);
                MPI_Recv(recv_buf.data(), num_cells * n_vars, MPI_DOUBLE, owner, nb, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                out.write(reinterpret_cast<const char*>(recv_buf.data()), recv_buf.size() * sizeof(double));
            }
        } else {
            if (owner == rank) {
                // 发送给 Rank 0
                // Tag 使用 block id (nb) 以防混淆
                MPI_Send(buffer.data(), buffer.size(), MPI_DOUBLE, 0, nb, MPI_COMM_WORLD);
            }
        }
        
        // 简单的同步，防止 IO 拥塞 (可选)
        // MPI_Barrier(MPI_COMM_WORLD); 
    }

    if (rank == 0) {
        out.close();
        spdlog::info("    Restart file written successfully.");
    }
}

// -------------------------------------------------------------------------
// 写单个 Block 的 VTS 文件 (Binary Appended, Interleaved Data)
// -------------------------------------------------------------------------
void NsKernel::write_block_vts(Block* b, const std::string& filename) {
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
void NsKernel::write_global_vtm(const std::string& filename, const std::string& step_str, int total_blocks) {
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

}