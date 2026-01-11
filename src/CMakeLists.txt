# ==========================================
# 1. 搜集源文件
# ==========================================

# 递归搜集所有的 .cpp 文件
file(GLOB_RECURSE SOURCES_CPP "*.cpp")

# 根据架构搜集设备代码
set(SOURCES_DEVICE "")

if(HYRES_ARCH STREQUAL "CUDA")
    # 搜集 .cu 文件
    file(GLOB_RECURSE SOURCES_DEVICE "solver/kernels/*.cu")
elseif(HYRES_ARCH STREQUAL "DCU")
    # TODO
endif()

# ==========================================
# 2. 定义可执行文件
# ==========================================

add_executable(hyres-solver
    ${SOURCES_CPP}
    ${SOURCES_DEVICE}
)

# ==========================================
# 3. 链接库与包含路径
# ==========================================

# 链接 MPI
target_link_libraries(hyres-solver PRIVATE MPI::MPI_CXX)

# 包含 src 根目录，这样代码里可以写 #include "common/Types.h"
target_include_directories(hyres-solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 设置编译宏 SPDLOG_ACTIVE_LEVEL，确保 Release 模式下不打印 Debug 信息
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(hyres-solver PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
else()
    target_compile_definitions(hyres-solver PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()

# 针对不同架构的特殊链接设置
if(HYRES_ARCH STREQUAL "DCU")
    # TODO
endif()