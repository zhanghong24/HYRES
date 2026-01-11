#pragma once

#include <vector>
#include "data/Block.h"
#include "common/Config.h" // 假设你的 Config 定义在这里
#include "data/GlobalData.h"

namespace Hyres {

class BoundaryManager {
public:
    // 构造函数：注入配置和块列表的引用
    BoundaryManager(const Config* config, const std::vector<Block*>& blocks);
    
    // 对外唯一接口：执行所有块的边界填充
    void boundary_sequence(Block* b);

private:
    // 成员变量
    const Config* config_;
    const std::vector<Block*>& blocks_; // 需要访问所有块以处理 Block-to-Block 连接

    // =========================================================
    // 内部实现函数
    // =========================================================

    // MPI / 对接边界 (Type < 0)
    void boundary_n1_vir3_parallel(Block* b, BoundaryPatch& patch);
    void boundary_n1_vir3(Block* b, BoundaryPatch& patch);
    void boundary_n1_vir3_other(Block* b, BoundaryPatch& patch);

    // 物理边界 (Type > 0)
    void boundary_symmetry(Block* b, const BoundaryPatch& patch);      // Type 1
    void boundary_viscid_wall(Block* b, const BoundaryPatch& patch);   // Type 2 (NS)
    void boundary_inviscid_wall(Block* b, const BoundaryPatch& patch); // Type 2 (Euler)
    void boundary_outflow(Block* b, const BoundaryPatch& patch);       // Type 6
    void boundary_farfield(Block* b, const BoundaryPatch& patch);      // Type 4
    void boundary_freestream(Block* b, const BoundaryPatch& patch);    // Type 5

    // 辅助工具
    void dif_average(Block* b, const BoundaryPatch& patch);
};

} // namespace Hyres