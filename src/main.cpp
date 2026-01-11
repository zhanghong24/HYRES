#include "common/MpiContext.h"
#include "common/Config.h"
#include "pre/SimulationLoader.h"
#include "solver/SolverDriver.h"
#include "spdlog/spdlog.h"

using namespace Hyres;

int main(int argc, char** argv) {

    spdlog::set_pattern("[%H:%M:%S] [%l] %v");

    // ========================================================
    // 1. 初始化 MPI 环境
    // ========================================================

    MpiContext mpi(&argc, &argv);

    // ========================================================
    // 2. 读取配置 (Input)
    // ========================================================

    Config *config = new Config;

    config->load("config.json", mpi.is_root());

    if (mpi.is_root()) { config->log_config(); }

    // ========================================================
    // 3. 前处理 (Preprocess)
    // ========================================================

    std::vector<Block*> blocks = SimulationLoader::load(config, mpi);

    // ========================================================
    // 4. 核心循环部分 (Solver)
    // ========================================================

    SolverDriver* driver = new SolverDriver(blocks, config, mpi);
    driver->solve();

    // ========================================================
    // 5. 结束
    // ========================================================
    for (auto* b : blocks) delete b;
    return 0;
}