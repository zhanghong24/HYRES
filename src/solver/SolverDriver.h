#pragma once

#include <vector>
#include <memory>
#include <iomanip> 
#include <algorithm>
#include "data/Block.h"
#include "common/Config.h"
#include "common/MpiContext.h"
#include "spdlog/spdlog.h"

namespace Hyres {

template <typename Kernel>
class SolverDriver {
public:
    // 构造函数：接收初始化好的 Block 列表和配置
    SolverDriver(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi)
        : blocks_(blocks), config_(config), mpi_(mpi), kernel_(blocks, config, mpi) {
        
        current_step_ = config->control.start_mode == 0 ? 0 : 0; // TODO: Handle restart step
        max_steps_ = config->control.max_steps;
    }
    
    // 析构函数
    ~SolverDriver() = default;

    // 核心入口：开始主循环
    void solve() {
        if (mpi_.is_root()) {
            spdlog::info(">>> Solver Started. Max Steps: {}", max_steps_);
        }

        // 主时间步循环 (Main Loop)
        for (int step = current_step_ + 1; step <= max_steps_; ++step) {
            
            // 1. 执行单步时间推进 (RK3, LU-SGS, etc.)
            run_time_step(step);

            // 2. 监控残差 (Residual Monitor)
            if (step % config_->control.res_interval == 0) {
                kernel_.check_residual(step);
            }

            // 3. I/O 输出 (Result Saving)
            if (step % config_->control.save_interval == 0) {
                kernel_.output_solution(step);
            }
        }

        if (mpi_.is_root()) {
            spdlog::info(">>> Solver Finished Successfully.");
        }
    }

private:
    void run_time_step(int step) {

        // 1. 填充 Ghost Cell
        kernel_.apply_boundary();

        // 2. 计算时间步长
        kernel_.compute_time_step();

        // 3. 计算总残差
        kernel_.compute_rhs();

        // 4. 时间推进部分
        kernel_.compute_lhs();
    }
    
private:
    std::vector<Block*>& blocks_; 
    const Config* config_;
    const MpiContext& mpi_;

    int current_step_;
    int max_steps_;

    Kernel kernel_;
};

} // namespace Hyres