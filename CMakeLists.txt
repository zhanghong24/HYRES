cmake_minimum_required(VERSION 3.18)

# 项目名称与版本
project(HYRES 
    VERSION 0.1.0 
    LANGUAGES CXX C
)

# ==========================================
# [新增] 默认构建模式设置为 Release
# ==========================================
# 如果用户没有通过 -DCMAKE_BUILD_TYPE=... 指定模式，则默认为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
    message(STATUS ">>> CMAKE_BUILD_TYPE not set, defaulting to Release mode.")
endif()

message(STATUS ">>> Current Build Type: ${CMAKE_BUILD_TYPE}")

# ==========================================
# 1. 全局配置与宏定义控制
# ==========================================

# 设置 C++ 标准 (C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 架构选择变量 (默认 CPU)
set(HYRES_ARCH "CPU" CACHE STRING "Target Architecture: CPU, CUDA, DCU")

message(STATUS ">>> Building HYRES for architecture: ${HYRES_ARCH}")

# 精度选择 (默认为 ON，即双精度)
option(HYRES_ENABLE_DOUBLE "Use double precision" ON)

if(HYRES_ENABLE_DOUBLE)
    add_compile_definitions(HYRES_USE_DOUBLE)
    message(STATUS ">>> Precision: DOUBLE")
else()
    message(STATUS ">>> Precision: SINGLE (Float)")
endif()

# ==========================================
# [关键] 针对 Release 模式的优化选项
# ==========================================
if(HYRES_ARCH STREQUAL "CPU")
    add_compile_definitions(HYRES_USE_CPU)
    
    # 这里定义了 Release 模式下的专用 Flag
    # -O3: 最高级别通用优化
    # -march=native: 针对当前 CPU 指令集优化 (如 AVX2/AVX512)
    # -funroll-loops: 循环展开
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
    
    # 如果您想确保 Debug 模式也不至于太慢，可以给 Debug 加个 -O1 或 -Og
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

elseif(HYRES_ARCH STREQUAL "CUDA")
    add_compile_definitions(HYRES_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    # CUDA 优化选项
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

elseif(HYRES_ARCH STREQUAL "DCU")
    # TODO
endif()

# ==========================================
# 2. 依赖库查找
# ==========================================

# MPI (必须)
find_package(MPI REQUIRED)
message(STATUS ">>> MPI Found: ${MPI_CXX_COMPILER}")

# 包含 external 目录
include_directories(${CMAKE_SOURCE_DIR}/external)

# ==========================================
# 3. 输出目录设置
# ==========================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ==========================================
# 4. 添加子目录
# ==========================================
add_subdirectory(src)