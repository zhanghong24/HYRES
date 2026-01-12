#include "NSKernel.h"

namespace Hyres {

// =========================================================================
// 核心函数：exchange_bc (修正接口调用)
// =========================================================================
void NsKernel::exchange_bc() {
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
// 全局 DQ 通信与归一化 (修复索引排序问题)
// =========================================================================
void NsKernel::communicate_dq_npp() {
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

    // 2. 对接面平滑 (2 Point Match)
    boundary_match_dq_2pm();

    // 3. 多点平均 (3 Point Plus)
    boundary_match_dq_3pp();
}

// =========================================================================
// 多点平均处理子函数 (Strictly following Fortran logic)
// 对应 Fortran: subroutine boundary_match_dq_3pp
// =========================================================================
void NsKernel::boundary_match_dq_3pp() {
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
void NsKernel::communicate_pv_npp() {
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
void NsKernel::boundary_match_pv_3pp() {
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
void NsKernel::exchange_bc_dq_vol() {
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
void NsKernel::exchange_bc_pv_vol() {
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
void NsKernel::boundary_match_pv_2pm() {
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
void NsKernel::boundary_match_dq_2pm() {
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

}