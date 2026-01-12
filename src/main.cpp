#include "common/MpiContext.h"
#include "common/Config.h"
#include "pre/SimulationLoader.h"
#include "solver/SolverDriver.h"
#include "solver/ns/NSKernel.h"
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
    // 4. 核心循环部分 (Solver Dispatch)
    // 根据 model_mode 进行静态分发
    // ========================================================
    
    // 0: NS/Euler (Laminar), 1: RANS (Turbulence), 2: NEMO (Chem/Noneq)
    int model_mode = config->physics.model_mode;

    if (model_mode == 0) {
        // --- Branch 0: Navier-Stokes / Euler Framework ---
        if (mpi.is_root()) {
            std::string sub_mode = (config->physics.viscous_mode == 0) ? "Euler" : "Laminar NS";
            spdlog::info(">>> Solver Mode: NS Kernel ({})", sub_mode);
        }

        auto* driver = new SolverDriver<NsKernel>(blocks, config, mpi);
        driver->solve();
        delete driver;
    }
    else if (model_mode == 1) {
        // --- Branch 1: RANS Framework (Placeholder) ---
        if (mpi.is_root()) {
            spdlog::info(">>> Solver Mode: RANS Kernel");
            spdlog::warn(">>> RANS module is currently under development.");
        }
        
        // TODO: Uncomment when RansKernel is ready
        // auto* driver = new SolverDriver<RansKernel>(blocks, config, mpi);
        // driver->solve();
        // delete driver;
        
        MPI_Abort(MPI_COMM_WORLD, 0); 
    }
    else if (model_mode == 2) {
        // --- Branch 2: NEMO Framework (Placeholder) ---
        if (mpi.is_root()) {
            spdlog::info(">>> Solver Mode: NEMO Kernel (Non-Equilibrium/Chem)");
            spdlog::warn(">>> NEMO module is currently under development.");
        }

        // TODO: Uncomment when NemoKernel is ready
        // auto* driver = new SolverDriver<NemoKernel>(blocks, config, mpi);
        // driver->solve();
        // delete driver;

        MPI_Abort(MPI_COMM_WORLD, 0); 
    }
    else {
        // --- Invalid Mode ---
        if (mpi.is_root()) {
            spdlog::critical(">>> Unknown physics.model_mode: {}. Valid options: 0 (NS), 1 (RANS), 2 (NEMO).", model_mode);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ========================================================
    // 5. 结束
    // ========================================================
    for (auto* b : blocks) delete b;
    delete config;

    return 0;
}