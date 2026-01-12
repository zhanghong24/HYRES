#include "NSKernel.h"
#include <fstream>

namespace Hyres {

NsKernel::NsKernel(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi) 
    : blocks_(blocks), config_(config), mpi_(mpi), boundary_manager_(config, blocks) {}

NsKernel::~NsKernel() = default;

void NsKernel::check_residual(int step) {
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
                     step, rms_residual, global_max_res.val, loc_str);
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