#include "common/MpiContext.h"
#include "spdlog/spdlog.h"
#include <vector>

namespace Hyres {

// 对应 Fortran 的 subroutine parallel_init
MpiContext::MpiContext(int* argc, char*** argv) {
    int ierr;
    
    // 1. 初始化 MPI
    // 在 Types.h 中定义的 MPI_REAL_T 已经编译时确定。
    MPI_Init(argc, argv);

    // 2. 获取 Rank 和 Size
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 3. 获取 Processor Name (对应 call MPI_GET_PROCESSOR_NAME)
    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);
    hostname = std::string(name, len);

    // 4. 配置日志 (如果是 Rank 0)
    if (rank == 0) {
        spdlog::info(">>> MPI Initialized. Total Ranks: {}", size);
    }
}

MpiContext::~MpiContext() {
    if (rank == 0) {
        spdlog::info(">>> MPI Finalizing...");
    }
    MPI_Finalize();
}

// Rank 0 收集所有人的主机名并打印，防止输出乱序
void MpiContext::log_topology() const {
    // 简单的同步，确保大家都到了
    barrier();

    // 如果只是简单的打印，可能会乱序。
    // 这里我们只让每个人打印自己的信息，依靠 spdlog 的格式化
    spdlog::info("[Rank {}/{}] is running on host: {}", rank, size, hostname);
    
    // 再次同步
    barrier();
}

} // namespace Hyres