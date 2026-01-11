#pragma once

#include <cmath>
#include <cstdint>

// ==========================================
// 1. 精度控制
// ==========================================

// 在 CMake 中通过 add_compile_definitions(HYRES_USE_DOUBLE) 来控制
#ifdef HYRES_USE_DOUBLE
    using real_t = double;
    #define MPI_REAL_T MPI_DOUBLE
#else
    using real_t = float;
    #define MPI_REAL_T MPI_FLOAT
#endif

// 整数类型 (用 int32_t 或 int64_t 明确位宽，防止不同平台歧义)
using idx_t = int32_t; 

// ==========================================
// 2. 物理常量
// ==========================================
namespace Hyres {

    constexpr real_t PI = 3.14159265358979323846;
    constexpr real_t SMALL_EPS = 1.0e-38; // 防止除以零的小量

} // namespace Hyres

// ==========================================
// 3. CUDA 兼容性宏
// ==========================================
#ifdef __CUDACC__
    #define HYRES_HOST_DEVICE __host__ __device__
    #define HYRES_DEVICE __device__
    #define HYRES_HOST __host__
#else
    #define HYRES_HOST_DEVICE 
    #define HYRES_DEVICE 
    #define HYRES_HOST 
#endif