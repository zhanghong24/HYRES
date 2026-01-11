#include "pre/SimulationLoader.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <sstream>   // std::ostringstream
#include <iomanip>   // std::setw

namespace Hyres {

// =========================================================================
// 主入口：Load (顺序已修正)
// =========================================================================
std::vector<Block*> SimulationLoader::load(const Config* config, const MpiContext& mpi) {
    std::vector<Block*> blocks;
    int ng = (config->scheme.scheme_id == 41) ? 3 : 0;

    // 1-2. Master 读取 (保持不变)
    if (mpi.is_root()) {
        spdlog::info(">>> Reading Grid: {}", config->files.grid);
        read_plot3d_grid(blocks, config->files.grid, ng);
        spdlog::info(">>> Reading BC: {}", config->files.bc);
        read_bc_parallel(blocks, config->files.bc, mpi);
    }

    // 3. 分发 Grid (Master 保留 Block 对象)
    if (mpi.is_root()) spdlog::info(">>> Distributing Grid...");
    distribute_grid(blocks, mpi, ng);

    // 4. 分发 BC 拓扑 (Slave 补全 Block 壳)
    if (mpi.is_root()) spdlog::info(">>> Distributing BC Topology...");
    distribute_bc(blocks, mpi, ng);

    // 5. 本地拓扑分析 (需要 Block 壳存在)
    if (mpi.is_root()) spdlog::info(">>> Analyzing BC Topology...");
    analyze_bc(blocks, mpi);

    // 6. 获取本地信息 (Log)
    get_local_info(blocks, mpi);

    // 7. 分配通信缓冲区 (Alloc Comm Arrays)
    if (mpi.is_root()) spdlog::info(">>> Allocating Comm Buffers...");
    alloc_bc_comm_array(blocks, mpi);

    // 8. Master 清理 (Cleanup)
    cleanup_blocks(blocks, mpi);

    // 9. 设置边界的执行顺序
    set_bc_index(blocks);
    if (mpi.is_root()) spdlog::info(">>> BC Execution Index built successfully.");

    // 10. 申请FlowSolution相关的变量内存
    allocate_other_variable(blocks, mpi);

    // 11. 计算mb的metrics & jacobian
    set_grid_derivative(blocks, mpi);

    // 12. 检查mb的metrics & jacobian计算质量
    check_grid_derivative(blocks, mpi);

    // 13. 检查负载均衡
    check_load_balance(blocks, mpi);

    // 14. 初始化无量纲来流条件
    PhysicsInitializer::init_inflow(config, mpi.get_rank());

    // 15. 初始化流场的原始量（无量纲）
    initialization(blocks, config, mpi);

    // 16. 建立多点索引关系
    build_nppos_list_ex(blocks, mpi);

    // 17. 重新计算全场的温度分布
    initialize_temperature(blocks, config, mpi);

    if (mpi.is_root()) {
        spdlog::info(">>> All Pre-processing Done.");
    }

    return blocks;
}

void SimulationLoader::read_plot3d_grid(std::vector<Block*>& blocks, 
                                        const std::string& filename, 
                                        int ng) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::error("Failed to open grid file: {}", filename);
        exit(EXIT_FAILURE);
    }

    // ---------------------------------------------------------
    // 1. 读取块数量 (read(101) nblocks)
    // ---------------------------------------------------------
    int nblocks = 0;
    read_int(ifs, nblocks);
    spdlog::info("    Number of blocks: {}", nblocks);

    GlobalData::getInstance().n_blocks = nblocks;

    if (nblocks <= 0) {
        spdlog::error("Invalid nblocks: {}", nblocks);
        exit(EXIT_FAILURE);
    }

    // ---------------------------------------------------------
    // 2. 读取每个块的维度 (do nb=1,nblocks; read...; end do)
    // ---------------------------------------------------------
    struct Dim3 { int ni, nj, nk; };
    std::vector<Dim3> dims(nblocks);
    
    for (int i = 0; i < nblocks; ++i) {
        int idim, jdim, kdim;
        read_3ints(ifs, idim, jdim, kdim);
        
        dims[i] = {idim, jdim, kdim};
        
        // 顺便在这里就把 Block 对象创建出来
        // Block ID 对应 i (从 0 开始，Fortran 是 1 开始)
        // 注意：这里 Rank 暂时填 0，因为目前所有块都在 Master 手里
        Block* b = new Block(i, idim, jdim, kdim, ng);
        blocks.push_back(b);
    }

    // ---------------------------------------------------------
    // 3. 读取每个块的坐标数据 (do nb=1,nblocks; read X,Y,Z; end do)
    // ---------------------------------------------------------
    for (int i = 0; i < nblocks; ++i) {
        int ni = dims[i].ni;
        int nj = dims[i].nj;
        int nk = dims[i].nk;
        size_t num_points = (size_t)ni * nj * nk;

        // 准备临时 Buffer (纯物理网格，不带 Ghost)
        std::vector<double> raw_x(num_points);
        std::vector<double> raw_y(num_points);
        std::vector<double> raw_z(num_points);

        // Fortran: read(101) x, y, z
        // 这是一个巨大的 Record，包含 3 个数组
        
        // 1. 跳过头部 (4 bytes)
        skip_marker(ifs);

        // 2. 读取 X, Y, Z (连续读取)
        // 注意：ifs.read 接收 char*，需要转换指针
        // 还要注意大小：num_points * sizeof(double)
        ifs.read(reinterpret_cast<char*>(raw_x.data()), num_points * sizeof(double));
        ifs.read(reinterpret_cast<char*>(raw_y.data()), num_points * sizeof(double));
        ifs.read(reinterpret_cast<char*>(raw_z.data()), num_points * sizeof(double));

        // 3. 跳过尾部 (4 bytes)
        skip_marker(ifs);

        // -----------------------------------------------------
        // 4. 调用 Block 的接口进行填充
        // -----------------------------------------------------
        // 这里会自动处理 Ghost Layer 偏移，把数据放在中心
        blocks[i]->fill_grid_data(raw_x, raw_y, raw_z);
    }

    ifs.close();
}

void SimulationLoader::read_bc_parallel(std::vector<Block*>& blocks, 
                                        const std::string& filename,
                                        const MpiContext& mpi) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        spdlog::error("Failed to open BC file: {}", filename);
        exit(EXIT_FAILURE);
    }

    int flow_solver_id;
    ifs >> flow_solver_id; 
    
    int nprocs_file;
    ifs >> nprocs_file; 

    int ntms = -1;
    int numprocs = mpi.get_size();

    if (nprocs_file > 0 && (nprocs_file % numprocs == 0)) {
        ntms = nprocs_file / numprocs;
    } else {
        ntms = -1;
    }
    
    spdlog::info("    BC Load Balance: File_Procs={}, Current_Procs={}, Scaling(ntms)={}", 
                 nprocs_file, numprocs, ntms);

    int number_of_blocks;
    ifs >> number_of_blocks;
    
    if (number_of_blocks != (int)blocks.size()) {
        spdlog::error("BC file block count mismatch!");
        exit(EXIT_FAILURE);
    }

    for (int nb = 0; nb < number_of_blocks; ++nb) {
        Block* block = blocks[nb]; 

        int pid_file; 
        ifs >> pid_file;

        int pid_final = pid_file;
        if (ntms > 0) {
            pid_final = (pid_file - 1) / ntms + 1;
        }
        block->owner_rank = pid_final - 1; // 只有这里转为 0-based 供 MPI 使用

        int imax, jmax, kmax;
        ifs >> imax >> jmax >> kmax;

        int ndif = std::abs(imax - block->ni) + std::abs(jmax - block->nj) + std::abs(kmax - block->nk);
        if (ndif != 0) {
            spdlog::error("Block {} dim mismatch!", nb+1);
            exit(EXIT_FAILURE);
        }

        ifs >> block->name;

        int nrmax;
        ifs >> nrmax;
        block->boundaries.resize(nrmax);

        for (int nr = 0; nr < nrmax; ++nr) {
            BoundaryPatch& patch = block->boundaries[nr];
            patch.id = nr;

            // 【核心修改】直接存入 raw_is, raw_ie，不做任何数学运算
            ifs >> patch.raw_is[0] >> patch.raw_ie[0] 
                >> patch.raw_is[1] >> patch.raw_ie[1] 
                >> patch.raw_is[2] >> patch.raw_ie[2] 
                >> patch.type;

            patch.target_block = -1;
            patch.window_id = 0;
            // 初始化 target raw 数组，避免脏数据
            patch.raw_target_is = {0,0,0};
            patch.raw_target_ie = {0,0,0};

            if (patch.type < 0) {
                int nbt_file, ibcwin;

                // 【核心修改】直接存入 raw_target_is/ie
                ifs >> patch.raw_target_is[0] >> patch.raw_target_ie[0]
                    >> patch.raw_target_is[1] >> patch.raw_target_ie[1]
                    >> patch.raw_target_is[2] >> patch.raw_target_ie[2]
                    >> nbt_file
                    >> ibcwin;

                // 目标块 ID 转为 0-based，方便 C++ 数组索引 blocks[id]
                patch.target_block = nbt_file - 1;
                patch.window_id = ibcwin;
            }
        }
    }

    ifs.close();
}

// =========================================================================
// 核心函数：distribute_grid (修复 Slave owner_rank 问题)
// =========================================================================
void SimulationLoader::distribute_grid(std::vector<Block*>& blocks, const MpiContext& mpi, int ng) {
    int rank = mpi.get_rank();
    int nblocks = blocks.size();
    
    // 1. 广播块总数
    MPI_Bcast(&nblocks, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 2. 广播元数据
    std::vector<BlockGeomMeta> metas(nblocks);
    if (mpi.is_root()) {
        for (int i = 0; i < nblocks; ++i) {
            metas[i].id = blocks[i]->id;
            metas[i].ni = blocks[i]->ni;
            metas[i].nj = blocks[i]->nj;
            metas[i].nk = blocks[i]->nk;
            metas[i].owner_rank = blocks[i]->owner_rank;
        }
    }
    MPI_Bcast(metas.data(), nblocks * sizeof(BlockGeomMeta), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 3. Slave 初始化 (占位)
    if (!mpi.is_root()) {
        blocks.resize(nblocks, nullptr); 
        for (int i = 0; i < nblocks; ++i) {
            if (metas[i].owner_rank == rank) {
                blocks[i] = new Block(metas[i].id, metas[i].ni, metas[i].nj, metas[i].nk, ng);
                
                // 【修复】Slave 必须记住这个块是自己的
                blocks[i]->owner_rank = metas[i].owner_rank;
            } else {
                blocks[i] = nullptr; 
            }
        }
    }

    // 4. 数据分发循环
    for (int i = 0; i < nblocks; ++i) {
        const auto& meta = metas[i];
        int owner = meta.owner_rank;
        size_t packsize = (size_t)meta.ni * meta.nj * meta.nk;

        if (owner != 0) {
            // --- Master Sender ---
            if (mpi.is_root()) {
                Block* b = blocks[i]; 
                std::vector<double> buf_x(packsize), buf_y(packsize), buf_z(packsize);
                
                for(int k=0; k<meta.nk; ++k) {
                    for(int j=0; j<meta.nj; ++j) {
                        for(int ii=0; ii<meta.ni; ++ii) {
                            size_t idx = ii + meta.ni * j + meta.ni * meta.nj * k;
                            buf_x[idx] = b->x(ii + ng, j + ng, k + ng);
                            buf_y[idx] = b->y(ii + ng, j + ng, k + ng);
                            buf_z[idx] = b->z(ii + ng, j + ng, k + ng);
                        }
                    }
                }

                MPI_Send(buf_x.data(), packsize, MPI_DOUBLE, owner, meta.id * 3 + 0, MPI_COMM_WORLD);
                b->x.clear(); 

                MPI_Send(buf_y.data(), packsize, MPI_DOUBLE, owner, meta.id * 3 + 1, MPI_COMM_WORLD);
                b->y.clear(); 

                MPI_Send(buf_z.data(), packsize, MPI_DOUBLE, owner, meta.id * 3 + 2, MPI_COMM_WORLD);
                b->z.clear(); 
            }
            // --- Slave Receiver ---
            else if (rank == owner) {
                Block* b = blocks[i];
                if (b) {
                    std::vector<double> buf_x(packsize), buf_y(packsize), buf_z(packsize);
                    MPI_Recv(buf_x.data(), packsize, MPI_DOUBLE, 0, meta.id * 3 + 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(buf_y.data(), packsize, MPI_DOUBLE, 0, meta.id * 3 + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(buf_z.data(), packsize, MPI_DOUBLE, 0, meta.id * 3 + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    b->fill_grid_data(buf_x, buf_y, buf_z);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// =========================================================================
// 核心函数：distribute_bc (修复 Slave owner_rank 问题)
// =========================================================================
void SimulationLoader::distribute_bc(std::vector<Block*>& blocks, const MpiContext& mpi, int ng) {
    int nblocks = blocks.size();

    // 0. Slave 补全缺失的 Block 对象 (Shell)
    // 重新广播元数据
    std::vector<BlockGeomMeta> metas(nblocks);
    if (mpi.is_root()) {
        for(int i=0; i<nblocks; ++i) {
            metas[i].id = blocks[i]->id;
            metas[i].ni = blocks[i]->ni;
            metas[i].nj = blocks[i]->nj;
            metas[i].nk = blocks[i]->nk;
            metas[i].owner_rank = blocks[i]->owner_rank;
        }
    }
    MPI_Bcast(metas.data(), nblocks * sizeof(BlockGeomMeta), MPI_BYTE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < nblocks; ++i) {
        if (blocks[i] == nullptr) {
            blocks[i] = new Block(metas[i].id, metas[i].ni, metas[i].nj, metas[i].nk, ng);
            blocks[i]->x.clear();
            blocks[i]->y.clear();
            blocks[i]->z.clear();
            
            // 【修复】Slave 必须知道这个 Shell 归谁管
            blocks[i]->owner_rank = metas[i].owner_rank;
        } else {
            // 已经是本地块，也要确保 owner 正确 (双重保险)
            blocks[i]->owner_rank = metas[i].owner_rank;
        }
    }

    // 1. 广播 nregions
    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b = blocks[nb];
        int nregions = 0;

        if (mpi.is_root()) {
            nregions = b->boundaries.size();
        }

        MPI_Bcast(&nregions, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!mpi.is_root()) {
            b->boundaries.resize(nregions);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 2. 广播 Patch 详情
    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b = blocks[nb];
        int nregions = b->boundaries.size(); 

        for (int nr = 0; nr < nregions; ++nr) {
            if (mpi.is_root()) {
                b->boundaries[nr].block_id = nb + 1; 
            }
            MPI_Bcast(&(b->boundaries[nr]), sizeof(BoundaryPatch), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD); 
    }
    
    if (mpi.is_root()) {
        spdlog::info("    BC Topology Distributed (All-to-All).");
    }
}
// =========================================================================
// 新增函数：Master 清理
// =========================================================================
void SimulationLoader::cleanup_blocks(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    int count = 0;

    for (int i = 0; i < blocks.size(); ++i) {
        Block* b = blocks[i];
        if (b != nullptr) {
            // 如果块不属于当前进程
            if (b->owner_rank != rank) {
                b->free_payload();
                count++;
            }
        }
    }

    if (mpi.is_root()) {
        spdlog::info(">>> Master cleanup: cleared payload of {} non-local blocks (Topology kept).", count);
    }
}

// =========================================================================
// 核心函数：analyze_bc_connect (复刻 Fortran 旋转逻辑)
// =========================================================================
void SimulationLoader::analyze_bc_connect(BoundaryPatch& patch, int ndim) {
    std::array<int, 3> s_st = patch.raw_is;
    std::array<int, 3> s_ed = patch.raw_ie;
    std::array<int, 3> t_st = patch.raw_target_is;
    std::array<int, 3> t_ed = patch.raw_target_ie;
    
    int s_nd_f = patch.s_nd + 1; 
    int t_nd_f = 0; 

    std::array<int, 3> s_sign, t_sign, st_sign;
    int s_t_dirction[3][3] = {{0}}; 

    // 1. 确定符号和 Target 法向
    for(int m=0; m<3; ++m) {
        if (t_st[m] == t_ed[m]) t_nd_f = m + 1;
        
        int s_st_abs = std::abs(s_st[m]);
        int s_ed_abs = std::abs(s_ed[m]);
        s_sign[m] = 1;
        if ((m + 1) != s_nd_f) {
            if (s_st_abs > s_ed_abs) s_sign[m] = -1;
        }

        int t_st_abs = std::abs(t_st[m]);
        int t_ed_abs = std::abs(t_ed[m]);
        t_sign[m] = 1;
        if ((m + 1) != t_nd_f) {
             if (t_st_abs > t_ed_abs) t_sign[m] = -1;
        }
    }

    // 2. 计算旋转矩阵
    bool match = false;
    for(int m=0; m<3; ++m) {
        for(int n=0; n<3; ++n) {
            if ((m + 1) != s_nd_f && (n + 1) != t_nd_f) {
                int js1 = s_st[m];
                if (std::abs(js1) < std::abs(s_ed[m])) js1 = s_ed[m];

                int js2 = t_st[n];
                if (std::abs(js2) < std::abs(t_ed[n])) js2 = t_ed[n];

                // 利用 Raw 符号判断重叠
                if ( (long long)js1 * js2 > 0 ) {
                    s_t_dirction[n][m] = 1; 
                    s_t_dirction[t_nd_f-1][s_nd_f-1] = 1;
                    int rem_n = 3 - n - (t_nd_f - 1);
                    int rem_m = 3 - m - (s_nd_f - 1);
                    if (rem_n >= 0 && rem_n < 3 && rem_m >= 0 && rem_m < 3) {
                        s_t_dirction[rem_n][rem_m] = 1;
                    }
                    match = true;
                    goto match_found; 
                }
            }
        }
    }
    match_found:; 

    if (!match) {
        spdlog::warn("BC Analysis: No axis match found for Block {} Patch {}", patch.block_id, patch.id);
    }

    // 3. 计算 st_sign
    for(int m=0; m<3; ++m) {
        int co = 0;
        for(int n=0; n<3; ++n) {
            if(s_t_dirction[n][m] == 1) co = n;
        }
        st_sign[m] = t_sign[co] * s_sign[m];
    }

    // 4. 预计算映射表
    auto size = patch.get_size();
    int count_i = size[0];
    int count_j = size[1];
    int count_k = size[2];
    int total_cells = count_i * count_j * count_k;

    patch.image.resize(total_cells);
    patch.jmage.resize(total_cells);
    patch.kmage.resize(total_cells);

    int s_st_abs[3] = { std::abs(s_st[0]), std::abs(s_st[1]), std::abs(s_st[2]) };
    int t_st_abs[3] = { std::abs(t_st[0]), std::abs(t_st[1]), std::abs(t_st[2]) };

    int flat_idx = 0;

    for(int k_step = 0; k_step < count_k; ++k_step) {
        int k_curr = s_st_abs[2] + k_step * s_sign[2];
        int kdelt = (k_curr - s_st_abs[2]) * st_sign[2];

        for(int j_step = 0; j_step < count_j; ++j_step) {
            int j_curr = s_st_abs[1] + j_step * s_sign[1];
            int jdelt = (j_curr - s_st_abs[1]) * st_sign[1];

            for(int i_step = 0; i_step < count_i; ++i_step) {
                int i_curr = s_st_abs[0] + i_step * s_sign[0];
                int idelt = (i_curr - s_st_abs[0]) * st_sign[0];

                int co_i = s_t_dirction[0][0]*idelt + s_t_dirction[0][1]*jdelt + s_t_dirction[0][2]*kdelt;
                int co_j = s_t_dirction[1][0]*idelt + s_t_dirction[1][1]*jdelt + s_t_dirction[1][2]*kdelt;
                int co_k = s_t_dirction[2][0]*idelt + s_t_dirction[2][1]*jdelt + s_t_dirction[2][2]*kdelt;

                // 存入 0-based 结果
                patch.image[flat_idx] = (t_st_abs[0] + co_i) - 1;
                patch.jmage[flat_idx] = (t_st_abs[1] + co_j) - 1;
                patch.kmage[flat_idx] = (t_st_abs[2] + co_k) - 1;

                flat_idx++;
            }
        }
    }

    // 保存矩阵
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) patch.dir_matrix[i][j] = s_t_dirction[i][j];
        patch.st_sign[i] = st_sign[i];
    }
}


// =========================================================================
// 核心函数：analyze_bc (遍历所有块)
// =========================================================================
void SimulationLoader::analyze_bc(std::vector<Block*>& blocks, const MpiContext& mpi) {
    for (Block* b : blocks) {
        if (b == nullptr) continue; 

        for (auto& patch : b->boundaries) {
            // 确定法向
            for(int m=0; m<3; ++m) {
                int abs_st = std::abs(patch.raw_is[m]);
                int abs_ed = std::abs(patch.raw_ie[m]);
                
                if (abs_st == abs_ed) {
                    patch.s_nd = m; // 0-based
                    if (abs_st == 1) patch.s_lr = -1; else patch.s_lr = 1;
                }
            }

            // 对接分析
            if (patch.type < 0) {
                analyze_bc_connect(patch, 3);
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// =========================================================================
// 核心函数：get_local_info [Fortran: parallel.f90 Line 939]
// =========================================================================
void SimulationLoader::get_local_info(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    int local_count = 0;
    
    // 对应 Fortran 逻辑：统计归属权
    for (Block* b : blocks) {
        // Master 此时已清理非本地块，所以必须判空
        if (b != nullptr && b->owner_rank == rank) {
            local_count++;
        }
    }
}

// =========================================================================
// 核心函数：alloc_bc_comm_array (修复验证埋点触发逻辑)
// =========================================================================
void SimulationLoader::alloc_bc_comm_array(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    int nblocks = blocks.size();

    // 1. 确保 Owner 表最新
    std::vector<int> owners(nblocks);
    if (mpi.is_root()) {
        for(int i=0; i<nblocks; ++i) if(blocks[i]) owners[i] = blocks[i]->owner_rank;
    } else {
        for(int i=0; i<nblocks; ++i) if(blocks[i]) owners[i] = blocks[i]->owner_rank;
    }
    MPI_Bcast(owners.data(), nblocks, MPI_INT, 0, MPI_COMM_WORLD);

    // 用于控制只打印一次验证信息
    bool debug_printed = false;

    // 2. 遍历分配
    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b_src = blocks[nb]; 
        if (b_src == nullptr) continue; 

        int id_src = owners[nb]; 
        int nregions = b_src->boundaries.size();

        for (int nr = 0; nr < nregions; ++nr) {
            BoundaryPatch& patch = b_src->boundaries[nr];
            
            if (patch.type < 0) {
                int nbt = patch.target_block;
                int id_des = owners[nbt];

                if (id_src != id_des) {
                    
                    // --- A. 发送方分配 ---
                    if (rank == id_src) {
                        std::array<int, 3> ibeg = { std::abs(patch.raw_is[0]), std::abs(patch.raw_is[1]), std::abs(patch.raw_is[2]) };
                        std::array<int, 3> iend = { std::abs(patch.raw_ie[0]), std::abs(patch.raw_ie[1]), std::abs(patch.raw_ie[2]) };
                        int idir = patch.s_nd;
                        int inrout = patch.s_lr;

                        // Group 1: 原始尺寸
                        int ni = std::abs(ibeg[0] - iend[0]) + 1;
                        int nj = std::abs(ibeg[1] - iend[1]) + 1;
                        int nk = std::abs(ibeg[2] - iend[2]) + 1;

                        patch.dqv.resize(5, ni, nj, nk);
                        patch.qpv_t.resize(6, ni, nj, nk);
                        patch.dqv_t.resize(5, ni, nj, nk);

                        // Group 2: -5 扩展 (vol)
                        std::array<int, 3> ibeg5 = ibeg;
                        std::array<int, 3> iend5 = iend;
                        ibeg5[idir] -= 5 * std::max(inrout, 0);
                        iend5[idir] -= 5 * std::min(inrout, 0);
                        int ni5 = std::abs(ibeg5[0] - iend5[0]) + 1;
                        int nj5 = std::abs(ibeg5[1] - iend5[1]) + 1;
                        int nk5 = std::abs(ibeg5[2] - iend5[2]) + 1;

                        patch.vol.resize(ni5, nj5, nk5);

                        // Group 3: -4 扩展 + 切向 (qpv)
                        std::array<int, 3> ibeg4 = ibeg;
                        std::array<int, 3> iend4 = iend;
                        ibeg4[idir] -= 4 * std::max(inrout, 0);
                        iend4[idir] -= 4 * std::min(inrout, 0);

                        int mcyc[] = {0, 1, 2, 0, 1};
                        int dims[3] = {b_src->ni, b_src->nj, b_src->nk};

                        for (int m = 1; m <= 2; ++m) {
                            int idir2 = mcyc[idir + m];
                            if (ibeg4[idir2] > 1) ibeg4[idir2] -= 1;
                            if (iend4[idir2] < dims[idir2]) iend4[idir2] += 1;
                        }
                        int ni4 = std::abs(ibeg4[0] - iend4[0]) + 1;
                        int nj4 = std::abs(ibeg4[1] - iend4[1]) + 1;
                        int nk4 = std::abs(ibeg4[2] - iend4[2]) + 1;

                        patch.qpv.resize(ni4, nj4, nk4, 5);
                    }

                    // --- B. 接收方分配 ---
                    if (rank == id_des) {
                        int nrt = patch.window_id; // 1-based
                        int target_patch_idx = nrt - 1; 
                        Block* b_des = blocks[patch.target_block];
                        BoundaryPatch& target_patch = b_des->boundaries[target_patch_idx];

                        // 使用 Source 的几何信息
                        std::array<int, 3> ibeg = { std::abs(patch.raw_is[0]), std::abs(patch.raw_is[1]), std::abs(patch.raw_is[2]) };
                        std::array<int, 3> iend = { std::abs(patch.raw_ie[0]), std::abs(patch.raw_ie[1]), std::abs(patch.raw_ie[2]) };
                        int idir = patch.s_nd;
                        int inrout = patch.s_lr;

                        int ni = std::abs(ibeg[0] - iend[0]) + 1;
                        int nj = std::abs(ibeg[1] - iend[1]) + 1;
                        int nk = std::abs(ibeg[2] - iend[2]) + 1;

                        target_patch.dqvpack.resize(5, ni, nj, nk);
                        target_patch.qpvpack_t.resize(6, ni, nj, nk);
                        target_patch.dqvpack_t.resize(5, ni, nj, nk);

                        // -5 Layers
                        std::array<int, 3> ibeg5 = ibeg;
                        std::array<int, 3> iend5 = iend;
                        ibeg5[idir] -= 5 * std::max(inrout, 0);
                        iend5[idir] -= 5 * std::min(inrout, 0);
                        int ni5 = std::abs(ibeg5[0] - iend5[0]) + 1;
                        int nj5 = std::abs(ibeg5[1] - iend5[1]) + 1;
                        int nk5 = std::abs(ibeg5[2] - iend5[2]) + 1;

                        // -4 Layers + Tangential
                        std::array<int, 3> ibeg4 = ibeg;
                        std::array<int, 3> iend4 = iend;
                        ibeg4[idir] -= 4 * std::max(inrout, 0);
                        iend4[idir] -= 4 * std::min(inrout, 0);

                        int dims[3] = {b_src->ni, b_src->nj, b_src->nk}; 
                        int mcyc[] = {0, 1, 2, 0, 1};
                        for (int m = 1; m <= 2; ++m) {
                            int idir2 = mcyc[idir + m];
                            if (ibeg4[idir2] > 1) ibeg4[idir2] -= 1;
                            if (iend4[idir2] < dims[idir2]) iend4[idir2] += 1;
                        }
                        int ni4 = std::abs(ibeg4[0] - iend4[0]) + 1;
                        int nj4 = std::abs(ibeg4[1] - iend4[1]) + 1;
                        int nk4 = std::abs(ibeg4[2] - iend4[2]) + 1;

                        target_patch.qpvpack.resize(ni4, nj4, nk4, 5);
                    }
                }
            }
        }
    }
    
    if (mpi.is_root()) {
        spdlog::info("    BC Comm Arrays Allocated.");
    }
}

void SimulationLoader::set_bc_index(std::vector<Block*>& blocks) {
    // 1. 定义优先级列表: (/-1, 4, 5, 6, 2, 3/)
    const std::vector<int> priority_list = {-1, 4, 5, 6, 2, 3};

    for (Block* b : blocks) {
        if (b == nullptr) continue;

        int nrmax = b->boundaries.size();
        b->bc_execution_order.clear();
        b->bc_execution_order.reserve(nrmax);

        // 2. 按优先级排序 (Bucket Sort Logic)
        for (int idbc : priority_list) {
            for (int nr = 0; nr < nrmax; ++nr) {
                if (b->boundaries[nr].type == idbc) {
                    // 记录 Patch 的索引 (0-based)
                    b->bc_execution_order.push_back(nr);
                }
            }
        }

        // 3. 安全性校验 (Validation)
        if (b->bc_execution_order.size() != nrmax) {
            spdlog::error("Block {} Error: Found undefined BC types!", b->id);
            spdlog::error("  Total Patches: {}", nrmax);
            spdlog::error("  Matched Patches: {}", b->bc_execution_order.size());
            spdlog::error("  Allowed Types: -1, 4, 5, 6, 2, 3");
            
            // 打印出有问题的 Patch
            for (int nr = 0; nr < nrmax; ++nr) {
                int type = b->boundaries[nr].type;
                bool found = false;
                for (int p : priority_list) if (p == type) found = true;
                if (!found) {
                    spdlog::error("  -> Patch {} has unknown Type {}", nr, type);
                }
            }
            exit(EXIT_FAILURE);
        }
    }
}

// =========================================================================
// 核心函数：allocate_other_variable
// =========================================================================
void SimulationLoader::allocate_other_variable(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    int nblocks = blocks.size();
    int nl = 5; 

    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b = blocks[nb];
        if (b != nullptr && b->owner_rank == rank) {
            
            // 统一使用 ng=3
            int size_i = b->ni + 2 * b->ng;
            int size_j = b->nj + 2 * b->ng;
            int size_k = b->nk + 2 * b->ng;

            // 1. Metrics
            b->vol.resize(size_i, size_j, size_k);
            b->volt.resize(size_i, size_j, size_k);
            b->kcx.resize(size_i, size_j, size_k); b->kcy.resize(size_i, size_j, size_k); 
            b->kcz.resize(size_i, size_j, size_k); b->kct.resize(size_i, size_j, size_k);
            b->etx.resize(size_i, size_j, size_k); b->ety.resize(size_i, size_j, size_k); 
            b->etz.resize(size_i, size_j, size_k); b->ett.resize(size_i, size_j, size_k);
            b->ctx.resize(size_i, size_j, size_k); b->cty.resize(size_i, size_j, size_k); 
            b->ctz.resize(size_i, size_j, size_k); b->ctt.resize(size_i, size_j, size_k);

            // 2. Primitives
            b->r.resize(size_i, size_j, size_k);
            b->u.resize(size_i, size_j, size_k);
            b->v.resize(size_i, size_j, size_k);
            b->w.resize(size_i, size_j, size_k);
            b->p.resize(size_i, size_j, size_k);
            b->t.resize(size_i, size_j, size_k);
            b->c.resize(size_i, size_j, size_k);

            // 3. Conservative (SOA: nl at last dim in resize args for HyresArray i-fastest)
            b->q.resize(size_i, size_j, size_k, nl);
            b->dq.resize(size_i, size_j, size_k, nl);
            b->qnc.resize(size_i, size_j, size_k, nl);
            b->qmc.resize(size_i, size_j, size_k, nl);

            // 4. Viscosity & Aux
            b->visl.resize(size_i, size_j, size_k);
            b->vist.resize(size_i, size_j, size_k);
            b->vist.fill(0.0); // Fortran Logic

            b->sra.resize(size_i, size_j, size_k);
            b->srb.resize(size_i, size_j, size_k);
            b->src.resize(size_i, size_j, size_k);
            b->srva.resize(size_i, size_j, size_k);
            b->srvb.resize(size_i, size_j, size_k);
            b->srvc.resize(size_i, size_j, size_k);
            b->dtdt.resize(size_i, size_j, size_k);
            b->rhs1.resize(size_i, size_j, size_k);
            b->fs.resize(size_i, size_j, size_k);
            b->srs.resize(size_i, size_j, size_k);
            b->qke.resize(size_i, size_j, size_k);
            b->distance.resize(size_i, size_j, size_k);

            // 5. BC Flags
            b->bc_flags.resize(6);
            b->bc_flags[0].resize(1, b->nj, b->nk);
            b->bc_flags[1].resize(1, b->nj, b->nk);
            b->bc_flags[2].resize(b->ni, 1, b->nk);
            b->bc_flags[3].resize(b->ni, 1, b->nk);
            b->bc_flags[4].resize(b->ni, b->nj, 1);
            b->bc_flags[5].resize(b->ni, b->nj, 1);
        }
    }
    
    fill_bc_flag(blocks, mpi);

    if (mpi.is_root()) {
        spdlog::info("    Variable Arrays Allocated (SOA layout, ng=3).");
    }
}

// =========================================================================
// 核心函数：fill_bc_flag
// =========================================================================
void SimulationLoader::fill_bc_flag(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    int nblocks = blocks.size();

    for (int nb = 0; nb < nblocks; ++nb) {
        Block* b = blocks[nb];
        if (b != nullptr && b->owner_rank == rank) {
            
            if (b->bc_execution_order.empty()) {
                set_bc_index(blocks); 
            }

            int nregions = b->boundaries.size();
            for (int ib = 0; ib < nregions; ++ib) {
                // Priority Order
                int nr = b->bc_execution_order[ib]; 
                BoundaryPatch& patch = b->boundaries[nr];
                
                // Use ABS values
                std::array<int, 3> ibeg = { std::abs(patch.raw_is[0]), std::abs(patch.raw_is[1]), std::abs(patch.raw_is[2]) };
                std::array<int, 3> iend = { std::abs(patch.raw_ie[0]), std::abs(patch.raw_ie[1]), std::abs(patch.raw_ie[2]) };
                
                int idir = patch.s_nd;
                int inrout = patch.s_lr;
                int m = 2 * idir + (inrout == 1 ? 1 : 0);
                
                int k_start = ibeg[2] - 1;
                int k_end   = iend[2] - 1;
                int j_start = ibeg[1] - 1;
                int j_end   = iend[1] - 1;
                int i_start = ibeg[0] - 1;
                int i_end   = iend[0] - 1;

                for (int k = k_start; k <= k_end; ++k) {
                    for (int j = j_start; j <= j_end; ++j) {
                        for (int i = i_start; i <= i_end; ++i) {
                            int flag_i = (idir == 0) ? 0 : i;
                            int flag_j = (idir == 1) ? 0 : j;
                            int flag_k = (idir == 2) ? 0 : k;
                            b->bc_flags[m](flag_i, flag_j, flag_k) = nr;
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
// 驱动函数：set_grid_derivative
// =========================================================================
void SimulationLoader::set_grid_derivative(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    
    if (mpi.is_root()) spdlog::info(">>> Calculating Grid Derivatives (GCL)...");

    for (Block* b : blocks) {
        // 只处理本地块
        if (b != nullptr && b->owner_rank == rank) {
            grid_derivative_nb(b);
        }
    }
    
    if (mpi.is_root()) spdlog::info("    Grid Derivatives Calculated.");
}

// =========================================================================
// 单块驱动：grid_derivative_nb
// =========================================================================
void SimulationLoader::grid_derivative_nb(Block* b) {
    // 1. 计算空间导数 (GCL算法)
    compute_metrics_gcl(b);

    // 2. 初始化时间导数 (Grid Velocities) 为 0
    b->kct.fill(0.0);
    b->ett.fill(0.0);
    b->ctt.fill(0.0);
}

// =========================================================================
// 核心算法：compute_metrics_gcl (复刻 Visbal GCL)
// =========================================================================
void SimulationLoader::compute_metrics_gcl(Block* b) {
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;
    int ng = b->ng;

    // 确定最大维度用于分配 buffer
    int nmax = std::max({ni, nj, nk});

    // -----------------------------------------------------
    // 1. 分配临时大数组 (Heap Allocation)
    // -----------------------------------------------------
    // 这里的尺寸我们只分配物理尺寸 ni, nj, nk 即可？
    // 不，为了后续能直接写回 Block 的全尺寸数组(含Ghost)，dxyz 最好也对应全尺寸。
    // 但是中间计算只针对物理区。我们用全尺寸分配，但只填充物理区。
    int size_i = b->x.dims()[0]; // ni + 2*ng
    int size_j = b->x.dims()[1];
    int size_k = b->x.dims()[2];
    
    // dxyz 存储 18 个导数分量 (SOA: i-fastest, 18 at last dim)
    HyresArray<double> dxyz(size_i, size_j, size_k, 18); 
    // xkcetct 存储 3 个分量
    HyresArray<double> xkcetct(size_i, size_j, size_k, 3);

    // 1D Buffer (注意：只存储物理长度 nmax)
    // Fortran: q(n, nmax), q_half(n, 0:nmax)
    // C++ Layout in vector: [var0_0...var0_ni-1][var1_0...] (SoA style for cache)
    // 但是为了方便 value_half_node 处理，我们可以让 buffer 也是 SoA。
    // vector size: nvar * nmax
    std::vector<double> tem3(3 * nmax);       
    std::vector<double> temp3(3 * (nmax + 1)); 
    
    std::vector<double> temp6(6 * nmax);      
    std::vector<double> temp6_1(6 * (nmax+1));
    std::vector<double> dtemp6(6 * nmax);    

    // -----------------------------------------------------
    // 2. 第一阶段：计算 Jacobian 导数 (x_xi, ...)
    // -----------------------------------------------------
    
    // --- I 方向 (xi) ---
    // 物理循环: k=1..nk, j=1..nj (对应 C++ 0..nk-1 加上 ng 偏移)
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            // 提取物理 I 线条
            for (int i = 0; i < ni; ++i) {
                // 读取全尺寸数组 (含 Ghost)
                tem3[0 * ni + i] = b->x(i + ng, j + ng, k + ng);
                tem3[1 * ni + i] = b->y(i + ng, j + ng, k + ng);
                tem3[2 * ni + i] = b->z(i + ng, j + ng, k + ng);
            }

            value_half_node(3, nmax, ni, tem3, temp3);
            flux_dxyz(3, nmax, ni, temp3, tem3); // Result back to tem3

            // 存回 dxyz (物理区)
            for (int i = 0; i < ni; ++i) {
                dxyz(i + ng, j + ng, k + ng, 0) = tem3[0 * ni + i];
                dxyz(i + ng, j + ng, k + ng, 3) = tem3[1 * ni + i];
                dxyz(i + ng, j + ng, k + ng, 6) = tem3[2 * ni + i];
            }
        }
    }

    // --- J 方向 (eta) ---
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            for (int j = 0; j < nj; ++j) {
                tem3[0 * nj + j] = b->x(i + ng, j + ng, k + ng);
                tem3[1 * nj + j] = b->y(i + ng, j + ng, k + ng);
                tem3[2 * nj + j] = b->z(i + ng, j + ng, k + ng);
            }
            
            value_half_node(3, nmax, nj, tem3, temp3);
            flux_dxyz(3, nmax, nj, temp3, tem3);

            for (int j = 0; j < nj; ++j) {
                dxyz(i + ng, j + ng, k + ng, 1) = tem3[0 * nj + j]; 
                dxyz(i + ng, j + ng, k + ng, 4) = tem3[1 * nj + j]; 
                dxyz(i + ng, j + ng, k + ng, 7) = tem3[2 * nj + j]; 
            }
        }
    }

    // --- K 方向 (zeta) ---
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            for (int k = 0; k < nk; ++k) {
                tem3[0 * nk + k] = b->x(i + ng, j + ng, k + ng);
                tem3[1 * nk + k] = b->y(i + ng, j + ng, k + ng);
                tem3[2 * nk + k] = b->z(i + ng, j + ng, k + ng);
            }
            
            value_half_node(3, nmax, nk, tem3, temp3);
            flux_dxyz(3, nmax, nk, temp3, tem3);

            for (int k = 0; k < nk; ++k) {
                dxyz(i + ng, j + ng, k + ng, 2) = tem3[0 * nk + k]; 
                dxyz(i + ng, j + ng, k + ng, 5) = tem3[1 * nk + k]; 
                dxyz(i + ng, j + ng, k + ng, 8) = tem3[2 * nk + k]; 
            }
        }
    }

    // -----------------------------------------------------
    // 3. 第二阶段：GCL 守恒 Metrics 计算
    // -----------------------------------------------------
    
    // 构造通量
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int ix = i + ng, iy = j + ng, iz = k + ng;
                
                // 备份 x_xi 等
                xkcetct(ix,iy,iz,0) = dxyz(ix,iy,iz,0);
                xkcetct(ix,iy,iz,1) = dxyz(ix,iy,iz,1);
                xkcetct(ix,iy,iz,2) = dxyz(ix,iy,iz,2);

                double xm = b->x(ix,iy,iz);
                double ym = b->y(ix,iy,iz);
                double zm = b->z(ix,iy,iz);

                b->kcx(ix,iy,iz) = dxyz(ix,iy,iz,0) * ym; 
                b->kcy(ix,iy,iz) = dxyz(ix,iy,iz,1) * ym; 
                b->kcz(ix,iy,iz) = dxyz(ix,iy,iz,2) * ym; 

                b->etx(ix,iy,iz) = dxyz(ix,iy,iz,3) * zm; 
                b->ety(ix,iy,iz) = dxyz(ix,iy,iz,4) * zm;
                b->etz(ix,iy,iz) = dxyz(ix,iy,iz,5) * zm;

                b->ctx(ix,iy,iz) = dxyz(ix,iy,iz,6) * xm; 
                b->cty(ix,iy,iz) = dxyz(ix,iy,iz,7) * xm;
                b->ctz(ix,iy,iz) = dxyz(ix,iy,iz,8) * xm;
            }
        }
    }

    // 对通量求导 (Differentiation)
    
    // I-Sweep
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            int ix = ng, iy = j + ng, iz = k + ng;
            for (int i = 0; i < ni; ++i) {
                temp6[0 * ni + i] = b->cty(i + ix, iy, iz);
                temp6[1 * ni + i] = b->ctz(i + ix, iy, iz);
                temp6[2 * ni + i] = b->kcy(i + ix, iy, iz);
                temp6[3 * ni + i] = b->kcz(i + ix, iy, iz);
                temp6[4 * ni + i] = b->ety(i + ix, iy, iz);
                temp6[5 * ni + i] = b->etz(i + ix, iy, iz);
            }
            value_half_node(6, nmax, ni, temp6, temp6_1);
            flux_dxyz(6, nmax, ni, temp6_1, dtemp6);
            
            for (int i = 0; i < ni; ++i) {
                for (int m = 0; m < 6; ++m) dxyz(i + ix, iy, iz, m) = dtemp6[m * ni + i];
            }
        }
    }

    // J-Sweep
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            int ix = i + ng, iy = ng, iz = k + ng;
            for (int j = 0; j < nj; ++j) {
                temp6[0 * nj + j] = b->ctx(ix, j + iy, iz);
                temp6[1 * nj + j] = b->ctz(ix, j + iy, iz);
                temp6[2 * nj + j] = b->kcx(ix, j + iy, iz);
                temp6[3 * nj + j] = b->kcz(ix, j + iy, iz);
                temp6[4 * nj + j] = b->etx(ix, j + iy, iz);
                temp6[5 * nj + j] = b->etz(ix, j + iy, iz);
            }
            value_half_node(6, nmax, nj, temp6, temp6_1);
            flux_dxyz(6, nmax, nj, temp6_1, dtemp6);
            
            for (int j = 0; j < nj; ++j) {
                for (int m = 0; m < 6; ++m) dxyz(ix, j + iy, iz, m + 6) = dtemp6[m * nj + j];
            }
        }
    }

    // K-Sweep
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            int ix = i + ng, iy = j + ng, iz = ng;
            for (int k = 0; k < nk; ++k) {
                temp6[0 * nk + k] = b->ctx(ix, iy, k + iz);
                temp6[1 * nk + k] = b->cty(ix, iy, k + iz);
                temp6[2 * nk + k] = b->kcx(ix, iy, k + iz);
                temp6[3 * nk + k] = b->kcy(ix, iy, k + iz);
                temp6[4 * nk + k] = b->etx(ix, iy, k + iz);
                temp6[5 * nk + k] = b->ety(ix, iy, k + iz);
            }
            value_half_node(6, nmax, nk, temp6, temp6_1);
            flux_dxyz(6, nmax, nk, temp6_1, dtemp6);
            
            for (int k = 0; k < nk; ++k) {
                for (int m = 0; m < 6; ++m) dxyz(ix, iy, k + iz, m + 12) = dtemp6[m * nk + k];
            }
        }
    }

    // 4. 最终组装
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int ix = i + ng, iy = j + ng, iz = k + ng;
                
                // Indices map: Fortran 1-based -> C++ 0-based
                // Fortran: 18, 12 -> C++ 17, 11
                b->kcx(ix,iy,iz) = dxyz(ix,iy,iz,17) - dxyz(ix,iy,iz,11);
                b->kcy(ix,iy,iz) = dxyz(ix,iy,iz,13) - dxyz(ix,iy,iz,7);
                b->kcz(ix,iy,iz) = dxyz(ix,iy,iz,15) - dxyz(ix,iy,iz,9);

                b->etx(ix,iy,iz) = dxyz(ix,iy,iz,5)  - dxyz(ix,iy,iz,16);
                b->ety(ix,iy,iz) = dxyz(ix,iy,iz,1)  - dxyz(ix,iy,iz,12);
                b->etz(ix,iy,iz) = dxyz(ix,iy,iz,3)  - dxyz(ix,iy,iz,14);

                b->ctx(ix,iy,iz) = dxyz(ix,iy,iz,10) - dxyz(ix,iy,iz,4);
                b->cty(ix,iy,iz) = dxyz(ix,iy,iz,6)  - dxyz(ix,iy,iz,0);
                b->ctz(ix,iy,iz) = dxyz(ix,iy,iz,8)  - dxyz(ix,iy,iz,2);

                // Vol
                b->vol(ix,iy,iz) = xkcetct(ix,iy,iz,0) * b->kcx(ix,iy,iz) + 
                                   xkcetct(ix,iy,iz,1) * b->etx(ix,iy,iz) + 
                                   xkcetct(ix,iy,iz,2) * b->ctx(ix,iy,iz);
            }
        }
    }
}

// =========================================================================
// 子函数：value_half_node (修正索引映射)
// =========================================================================
void SimulationLoader::value_half_node(int n, int nmax, int len, 
                                       const std::vector<double>& f, 
                                       std::vector<double>& fh) {
    // f layout: SOA [var0...][var1...]
    // fh layout: SOA [var0...][var1...]
    // len corresponds to ni.
    // f indices: 0..len-1 (nodes 1..ni)
    // fh indices: 0..len (faces 0.5..ni+0.5)
    
    int stride_f = len;
    int stride_fh = len + 1;

    for (int m = 0; m < n; ++m) {
        const double* q = &f[m * stride_f];
        double* q_half = &fh[m * stride_fh];

        // 1. 内部点 (Fortran: 2 to ni-2)
        // C++: i corresponds to Fortran i.
        // Fortran q_half(i) is face i+0.5.
        // Input q(i) is C++ q[i-1].
        // Formula: (9(q_i + q_i+1) - (q_i-1 + q_i+2))/16
        // C++ q[i] is q(i+1).
        // So for C++ index k (representing Fortran i):
        // q_half[k] uses q[k-1], q[k], q[k-2], q[k+1].
        
        for (int i = 2; i <= len - 2; ++i) {
            // q[i-1] is q(i), q[i] is q(i+1)
            q_half[i] = (9.0 * (q[i-1] + q[i]) - (q[i-2] + q[i+1])) / 16.0;
        }

        // 2. 左边界 (Fortran i=1) -> C++ index 1
        // Formula: (5q1 + 15q2 - 5q3 + q4)/16
        // C++ q[0] is q1.
        q_half[1] = (5.0*q[0] + 15.0*q[1] - 5.0*q[2] + q[3]) / 16.0;

        // 3. 右边界 (Fortran i=ni-1) -> C++ index len-1
        // Formula: (5q_ni + 15q_ni-1 - 5q_ni-2 + q_ni-3)/16
        // C++ q[len-1] is q_ni.
        q_half[len-1] = (5.0*q[len-1] + 15.0*q[len-2] - 5.0*q[len-3] + q[len-4]) / 16.0;

        // 4. 端点 (Fortran 0, ni) -> C++ 0, len
        
        // Left End (Face 0.5)
        // Formula: (35q1 - 35q2 + 21q3 - 5q4)/16
        q_half[0] = (35.0*q[0] - 35.0*q[1] + 21.0*q[2] - 5.0*q[3]) / 16.0;
        
        // Right End (Face ni+0.5)
        // Formula: (35q_ni - 35q_ni-1 + 21q_ni-2 - 5q_ni-3)/16
        q_half[len] = (35.0*q[len-1] - 35.0*q[len-2] + 21.0*q[len-3] - 5.0*q[len-4]) / 16.0;
    }
}

// =========================================================================
// 子函数：flux_dxyz (修复内部循环索引偏移)
// =========================================================================
void SimulationLoader::flux_dxyz(int n, int nmax, int len, 
                                 const std::vector<double>& fh, 
                                 std::vector<double>& df) {
    // fh: Input half-node values. Size: n * (len+1). Indices 0..len mapped from Fortran 0..ni.
    // df: Output derivatives. Size: n * len. Indices 0..len-1 mapped from Fortran 1..ni.
    
    int stride_fh = len + 1;
    int stride_df = len;

    for (int m = 0; m < n; ++m) {
        const double* f = &fh[m * stride_fh];
        double* d = &df[m * stride_df];

        // 1. Internal (Fortran 3..ni-2) -> C++ 2..len-3
        // Fortran: df(i) uses f(i)-f(i-1) ...
        // C++: d[i] corresponds to Fortran df(i+1).
        //      So we need Fortran f(i+1)-f(i).
        //      Mapped to C++ fh: f[i+1] - f[i].
        
        for (int i = 2; i <= len - 3; ++i) {
            d[i] = (2250.0 * (f[i+1] - f[i]) 
                  - 125.0  * (f[i+2] - f[i-1]) 
                  + 9.0    * (f[i+3] - f[i-2])) / 1920.0;
        }

        // 2. Near Boundary (Fortran 2, ni-1) -> C++ 1, len-2
        // f(2) = (f(0) - 27f(1) + 27f(2) - f(3)) / 24
        // Direct index mapping works here because coefficients are explicit
        d[1] = (f[0] - 27.0*f[1] + 27.0*f[2] - f[3]) / 24.0;
        
        // f(ni-1) = -(f(ni) - 27f(ni-1) + 27f(ni-2) - f(ni-3)) / 24
        // C++ f[len] is f(ni)
        d[len-2] = -(f[len] - 27.0*f[len-1] + 27.0*f[len-2] - f[len-3]) / 24.0;

        // 3. Boundary (Fortran 1, ni) -> C++ 0, len-1
        // f(1) = (-23f(0) + 21f(1) + 3f(2) - f(3)) / 24
        d[0] = (-23.0*f[0] + 21.0*f[1] + 3.0*f[2] - f[3]) / 24.0;

        // f(ni) = -(-23f(ni) + 21f(ni-1) + 3f(ni-2) - f(ni-3)) / 24
        d[len-1] = -(-23.0*f[len] + 21.0*f[len-1] + 3.0*f[len-2] - f[len-3]) / 24.0;
    }
}

// =========================================================================
// 核心函数：check_grid_derivative (几何质量检查 - 无极点处理版)
// =========================================================================
void SimulationLoader::check_grid_derivative(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();

    // --- [配置项] 请确认这些常数与 Fortran global_variables 一致 ---
    // Fortran: large, sml_vol
    const double large = 1.0e30;
    const double sml_vol = 1.0e-20; // 极小体积阈值
    // -------------------------------------------------------------

    double min_vol_local = large;
    double max_vol_local = -large;
    int neg_vol_local = 0;

    // 1. 遍历本地块进行处理
    for (Block* b : blocks) {
        if (b != nullptr && b->owner_rank == rank) {
            int ni = b->ni;
            int nj = b->nj;
            int nk = b->nk;
            int ng = b->ng;

            // 注意：这里不再处理 Type 71/72/73 的极点修正

            // --- 统计 Min/Max 并检查负体积 ---
            // 关键：只遍历物理区域 (1..ni -> 0..ni-1)
            int neg_vol_nb = 0; // 当前块的负体积计数
            
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        // 加上 ng 偏移访问物理区
                        double v = b->vol(i + ng, j + ng, k + ng);

                        // 更新统计
                        if (v < min_vol_local) min_vol_local = v;
                        if (v > max_vol_local) max_vol_local = v;

                        // 检查负体积
                        if (v < sml_vol) {
                            neg_vol_nb++;
                            neg_vol_local++;

                            // 打印前3个错误
                            if (neg_vol_nb <= 3) {
                                // 输出 1-based 索引以匹配 Fortran 日志习惯
                                spdlog::warn("Jacobi<0: Block {}, I={}, J={}, K={}, Vol={:e}", 
                                             b->id, i+1, j+1, k+1, v);
                            }

                            // 修正
                            b->vol(i + ng, j + ng, k + ng) = sml_vol;
                        }
                    }
                }
            }
        }
    }

    // 2. MPI 全局归约
    double min_vol_global, max_vol_global;
    int neg_vol_global;

    // 注意：min 初始化为 large，max 初始化为 -large，Reduce 后能得到正确的全局极值
    MPI_Reduce(&min_vol_local, &min_vol_global, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_vol_local, &max_vol_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&neg_vol_local, &neg_vol_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 3. 结果报告
    if (mpi.is_root()) {
        if (neg_vol_global > 0) {
            spdlog::error("ERROR(Jacobi<0): Found {} negative volumes! Stopping.", neg_vol_global);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            // 格式化输出，尽可能接近 Fortran 格式方便对比
            // Fortran: '$ Interval of Jacobi: (',minvol_t,',',maxvol_t,')'
            spdlog::info("$ Interval of Jacobi: ({:.5e}, {:.5e})", min_vol_global, max_vol_global);
        }
    }
}

// =========================================================================
// 新增函数：check_load_balance (负载均衡检测)
// =========================================================================
void SimulationLoader::check_load_balance(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    int size = mpi.get_size();
    
    // 1. 统计本地物理网格总数
    long long local_n_cells = 0;
    for (const auto* b : blocks) {
        // 只统计非空且属于自己的块
        if (b != nullptr && b->owner_rank == rank) {
            // 物理网格数 = ni * nj * nk (不包含 2*ng)
            local_n_cells += (long long)b->ni * b->nj * b->nk;
        }
    }

    // 2. 汇总到 Master
    std::vector<long long> gathered_cells;
    if (mpi.is_root()) {
        gathered_cells.resize(size);
    }

    // 使用 long long 防止网格数溢出 int
    MPI_Gather(&local_n_cells, 1, MPI_LONG_LONG, 
               gathered_cells.data(), 1, MPI_LONG_LONG, 
               0, MPI_COMM_WORLD);

    // 3. Master 计算并打印
    if (mpi.is_root()) {
        long long total_cells = 0;
        for (auto n : gathered_cells) total_cells += n;
        
        // 防止除以 0
        double avg_cells = (size > 0) ? ((double)total_cells / size) : 0.0;
        
        spdlog::info(">>> Load Balance Check");
        spdlog::info("    Total Cells: {}", total_cells);
        spdlog::info("    Avg per Rank: {:.0f}", avg_cells);
        
        for (int r = 0; r < size; ++r) {
            long long count = gathered_cells[r];
            double diff = (double)count - avg_cells;
            double percent = (avg_cells > 0) ? (diff / avg_cells * 100.0) : 0.0;
            
            // 格式化输出: Rank 0: Total 40000 cells +0.20%
            // 使用 {:+5.2f} 可以自动显示正负号
            spdlog::info("    Rank {}: Total {:<8} cells {:+.2f}%", r, count, percent);
        }
        spdlog::info("---------------------------------------------");
    }
}

// =========================================================================
// 驱动函数：initialization (流场初始化)
// =========================================================================
void SimulationLoader::initialization(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi) {
    // 对应 Fortran: if (nstart == 0) call init_nstart0 else call init_nstart1
    int nstart = config->control.start_mode;

    if (nstart == 0) {
        init_nstart0(blocks, mpi);
    } else {
        // [TODO] Restart mode (nstart=1)
        // 先用占位符，后续再实现 init_nstart1
        if (mpi.is_root()) {
            spdlog::warn(">>> [TODO] Restart mode (nstart={}) not implemented yet.", nstart);
        }
    }
}

// =========================================================================
// 核心函数：init_nstart0 (均匀流初始化 - 仅原始变量)
// =========================================================================
void SimulationLoader::init_nstart0(std::vector<Block*>& blocks, const MpiContext& mpi) {
    int rank = mpi.get_rank();
    auto& g = GlobalData::getInstance(); 

    for (Block* b : blocks) {
        // 只处理本地块
        if (b != nullptr && b->owner_rank == rank) {
            int ni = b->ni;
            int nj = b->nj;
            int nk = b->nk;
            int ng = b->ng;

            // 遍历物理区域 (Physical Domain)
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        int ix = i + ng;
                        int iy = j + ng;
                        int iz = k + ng;

                        // 1. 设置原始变量 (Primitive Variables)
                        b->r(ix, iy, iz) = g.rho_inf;  // roo
                        b->u(ix, iy, iz) = g.u_inf;    // uoo
                        b->v(ix, iy, iz) = g.v_inf;    // voo
                        b->w(ix, iy, iz) = g.w_inf;    // woo
                        b->p(ix, iy, iz) = g.p_inf;    // poo
                        b->t(ix, iy, iz) = 1.0;        // t=1.0
                    }
                }
            }
        }
    }

    if (mpi.is_root()) {
        spdlog::info(">>> Flow Field Initialized (nstart=0).");
    }
}

// =========================================================================
// 核心函数：initialize_temperature (对应 Fortran INITIAL_T & get_t_ini)
// =========================================================================
void SimulationLoader::initialize_temperature(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi) {
    // 对应 Fortran: if (nvis == 1) then ...
    // nvis 对应 config->physics.viscous_mode
    if (config->physics.viscous_mode != 1) {
        return;
    }

    int rank = mpi.get_rank();
    auto& g = GlobalData::getInstance();
    
    // 准备常量
    // Fortran: moo2 = moo * moo
    real_t gamma = g.gamma;
    real_t mach_sq = g.mach * g.mach;

    for (Block* b : blocks) {
        // 只处理本地块
        if (b != nullptr && b->owner_rank == rank) {
            int ni = b->ni;
            int nj = b->nj;
            int nk = b->nk;
            int ng = b->ng;

            // 遍历物理区域 (Physical Domain)
            // Fortran: do k=1,nk; do j=1,nj; do i=1,ni
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        int ix = i + ng;
                        int iy = j + ng;
                        int iz = k + ng;

                        real_t rho_val = b->r(ix, iy, iz); // rm
                        real_t p_val = b->p(ix, iy, iz);   // pm

                        // Fortran: a2 = gama * pm / rm
                        // Fortran: t(i,j,k) = moo2 * a2
                        if (rho_val > 1.0e-30) { // 简单防除零保护
                            real_t a2 = gamma * p_val / rho_val;
                            b->t(ix, iy, iz) = mach_sq * a2;
                        } else {
                            b->t(ix, iy, iz) = 0.0;
                        }
                    }
                }
            }
        }
    }

    if (mpi.is_root()) {
        spdlog::info(">>> Temperature Field Initialized (nvis=1).");
    }
}

// =========================================================================
// 内部辅助结构：并查集 (仅在 cpp 中可见)
// =========================================================================
namespace {
    struct PointKey {
        int block_id; int i, j, k;
        // 用于 map 的比较操作符
        bool operator<(const PointKey& o) const {
            return std::tie(block_id, i, j, k) < std::tie(o.block_id, o.i, o.j, o.k);
        }
        bool operator==(const PointKey& o) const {
            return std::tie(block_id, i, j, k) == std::tie(o.block_id, o.i, o.j, o.k);
        }
    };

    struct UnionFind {
        std::map<PointKey, PointKey> parent;
        
        PointKey find(PointKey p) {
            if (parent.find(p) == parent.end()) parent[p] = p;
            if (parent[p] == p) return p;
            return parent[p] = find(parent[p]);
        }

        void unite(PointKey p1, PointKey p2) {
            PointKey r1 = find(p1);
            PointKey r2 = find(p2);
            if (!(r1 == r2)) {
                // 简单的确定性合并规则
                if (r1 < r2) parent[r2] = r1;
                else parent[r1] = r2;
            }
        }
    };
}

void SimulationLoader::build_nppos_list_ex(std::vector<Block*>& blocks, const MpiContext& mpi) {
    auto& g = GlobalData::getInstance();
    int rank_size = mpi.get_size(); 
    int my_rank = mpi.get_rank();
    
    // 1. 初始化
    g.singularity_groups.clear();
    g.nppos_local.assign(rank_size, 0);
    g.ipp_st_local.assign(rank_size, 0);

    // 2. 并查集 (DSU) 识别连通点组
    // PointKey: {BlockID, i, j, k} (0-based)
    struct PointKey { 
        int b, i, j, k; 
        bool operator<(const PointKey& o) const {
            return std::tie(b,i,j,k) < std::tie(o.b,o.i,o.j,o.k);
        }
        bool operator==(const PointKey& o) const {
            return std::tie(b,i,j,k) == std::tie(o.b,o.i,o.j,o.k);
        }
    };

    std::map<PointKey, PointKey> parent;

    // 查找并路径压缩
    // 注意：不能使用递归 lambda，使用 std::function 或显式 Y-combinator
    // 这里使用简单的迭代查找
    auto find_set = [&](PointKey p) -> PointKey {
        // 如果点不在 map 中，将其添加为自身的根
        if (parent.find(p) == parent.end()) {
            parent[p] = p;
            return p;
        }

        PointKey root = p;
        while (!(parent[root] == root)) {
            root = parent[root];
        }

        // 路径压缩
        PointKey curr = p;
        while (!(curr == root)) {
            PointKey next = parent[curr];
            parent[curr] = root;
            curr = next;
        }
        return root;
    };

    auto union_sets = [&](PointKey a, PointKey b) {
        PointKey root_a = find_set(a);
        PointKey root_b = find_set(b);
        if (!(root_a == root_b)) {
            parent[root_a] = root_b;
        }
    };

    // 3. 遍历所有块的所有对接边界 (Type < 0)，建立连通关系
    // 注意：distribute_bc 保证了所有进程都有所有块的拓扑信息 (Boundaries)
    for (Block* b : blocks) {
        if (!b) continue;

        for (const auto& patch : b->boundaries) {
            if (patch.type >= 0) continue; // 只处理对接边界

            // 重建 analyze_bc_connect 中的遍历逻辑
            // 确保遍历顺序与 image/jmage/kmage 的填充顺序一致 (K, J, I loop)
            
            // 1. 获取原始范围和步长 (1-based, 带符号)
            int s_st[3] = {patch.raw_is[0], patch.raw_is[1], patch.raw_is[2]};
            int s_ed[3] = {patch.raw_ie[0], patch.raw_ie[1], patch.raw_ie[2]};
            
            int s_sign[3];
            for(int m=0; m<3; ++m) {
                s_sign[m] = (std::abs(s_st[m]) > std::abs(s_ed[m])) ? -1 : 1;
            }

            int count_i = std::abs(s_st[0] - s_ed[0]) + 1;
            int count_j = std::abs(s_st[1] - s_ed[1]) + 1;
            int count_k = std::abs(s_st[2] - s_ed[2]) + 1;

            int s_st_abs[3] = { std::abs(s_st[0]), std::abs(s_st[1]), std::abs(s_st[2]) };
            
            size_t flat_idx = 0;

            for(int k_step = 0; k_step < count_k; ++k_step) {
                int k_curr = s_st_abs[2] + k_step * s_sign[2]; // 1-based
                for(int j_step = 0; j_step < count_j; ++j_step) {
                    int j_curr = s_st_abs[1] + j_step * s_sign[1];
                    for(int i_step = 0; i_step < count_i; ++i_step) {
                        int i_curr = s_st_abs[0] + i_step * s_sign[0];

                        // 源点 (0-based)
                        PointKey u = { b->id, i_curr - 1, j_curr - 1, k_curr - 1 };

                        // 目标点 (0-based, image 数组中已存储 0-based)
                        int t_b = patch.target_block; // 0-based block id
                        int t_i = patch.image[flat_idx];
                        int t_j = patch.jmage[flat_idx];
                        int t_k = patch.kmage[flat_idx];

                        PointKey v = { t_b, t_i, t_j, t_k };

                        // 合并集合
                        union_sets(u, v);

                        flat_idx++;
                    }
                }
            }
        }
    }

    // 4. 收集连通分量 (Group Points)
    // Map: Root -> List of Points
    std::map<PointKey, std::vector<PointKey>> components;
    for (const auto& kv : parent) {
        PointKey root = find_set(kv.first);
        components[root].push_back(kv.first);
    }

    // 5. 过滤并构建 GlobalData (SingularityGroup > 2)
    // 对应 Fortran: if (current%nsimp_pnts > 2)
    for (const auto& kv : components) {
        if (kv.second.size() > 2) {
            SingularityGroup sg;
            sg.points.reserve(kv.second.size());
            for (const auto& p : kv.second) {
                // global_buffer_index 暂时填 0，稍后计算
                sg.points.push_back({p.b, p.i, p.j, p.k, 0});
            }
            g.singularity_groups.push_back(sg);
        }
    }

    // 6. 统计每个 Rank 负责的奇异点数量 (nppos_local)
    for (const auto& sg : g.singularity_groups) {
        for (const auto& p : sg.points) {
            int owner = blocks[p.block_id]->owner_rank;
            if (owner >= 0 && owner < rank_size) {
                g.nppos_local[owner]++;
            }
        }
    }

    // 7. 计算全局偏移 (Prefix Sum -> ipp_st_local)
    g.ipp_st_local[0] = 0;
    for (int r = 1; r < rank_size; ++r) {
        g.ipp_st_local[r] = g.ipp_st_local[r-1] + g.nppos_local[r-1];
    }

    // 8. 分配全局序列号并填充本地 Block 信息
    // 必须按 Rank 顺序分配 Index，以匹配 MPI_Allgatherv 的线性缓冲区顺序
    std::vector<int> current_counters = g.ipp_st_local; // Copy for increment

    for (auto& sg : g.singularity_groups) {
        for (auto& p : sg.points) {
            int owner = blocks[p.block_id]->owner_rank;
            
            // 分配全局 Buffer Index
            p.global_buffer_index = current_counters[owner]++;

            // 如果该点属于当前进程，记录到 Block 的本地奇异点列表中
            if (owner == my_rank) {
                Block* b = blocks[p.block_id];
                // 计算在本地发送 Buffer (dq_npp_local) 中的相对偏移
                int local_seq = p.global_buffer_index - g.ipp_st_local[owner];
                
                b->singular_points.push_back({p.i, p.j, p.k, local_seq});
            }
        }
    }

    // 9. 计算总数并分配全局通信 Buffer
    g.total_npp = 0;
    for (int c : g.nppos_local) g.total_npp += c;

    g.dq_npp_global.resize(5 * g.total_npp);
    g.pv_npp_global.resize(6 * g.total_npp);

    // 分配本地发送 Buffer
    int my_npp = g.nppos_local[my_rank];
    g.dq_npp_local.resize(5 * my_npp);
    g.pv_npp_local.resize(6 * my_npp);

    if (mpi.is_root()) {
        spdlog::info("$ multiple-points connection: {}", g.singularity_groups.size());
        spdlog::info("  Total Singular Points: {}", g.total_npp);
    }
}
} // namespace Hyres