#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <iostream>
#include "common/Types.h"

namespace Hyres {

class MpiContext {
public:
    // 构造函数
    MpiContext(int* argc, char*** argv);

    // 析构函数
    ~MpiContext();

    // 禁止拷贝
    MpiContext(const MpiContext&) = delete;
    MpiContext& operator=(const MpiContext&) = delete;

    // 获取当前进程信息
    int get_rank() const { return rank; }
    int get_size() const { return size; }
    bool is_root() const { return rank == 0; }
    
    // 全局同步
    void barrier() const { MPI_Barrier(comm); }

    // 打印拓扑信息
    void log_topology() const;

    // ========================================================
    // MPI 归约通信封装 (Allreduce)
    // 修复：使用 helper 函数重载代替模板特化，解决编译错误
    // ========================================================

    // 1. Min
    template <typename T>
    T allreduce_min(T local_val) const {
        T global_val;
        // 通过传入 local_val 让编译器自动匹配对应的重载函数
        MPI_Allreduce(&local_val, &global_val, 1, get_mpi_data_type_helper(local_val), MPI_MIN, comm);
        return global_val;
    }

    // 2. Max
    template <typename T>
    T allreduce_max(T local_val) const {
        T global_val;
        MPI_Allreduce(&local_val, &global_val, 1, get_mpi_data_type_helper(local_val), MPI_MAX, comm);
        return global_val;
    }

    // 3. Sum
    template <typename T>
    T allreduce_sum(T local_val) const {
        T global_val;
        MPI_Allreduce(&local_val, &global_val, 1, get_mpi_data_type_helper(local_val), MPI_SUM, comm);
        return global_val;
    }

private:
    int rank = 0;
    int size = 1;
    MPI_Comm comm = MPI_COMM_WORLD;
    std::string hostname;

    // ========================================================
    // 内部辅助：类型映射 (使用静态重载，编译器自动选择)
    // ========================================================
    static MPI_Datatype get_mpi_data_type_helper(double) { return MPI_DOUBLE; }
    static MPI_Datatype get_mpi_data_type_helper(float)  { return MPI_FLOAT; }
    static MPI_Datatype get_mpi_data_type_helper(int)    { return MPI_INT; }
    static MPI_Datatype get_mpi_data_type_helper(long)   { return MPI_LONG; }
};

} // namespace Hyres