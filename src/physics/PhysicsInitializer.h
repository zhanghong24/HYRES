#pragma once

#include "common/Types.h"
#include "common/Config.h"
#include "data/GlobalData.h"

namespace Hyres {

class PhysicsInitializer {
public:
    // 主入口：初始化流场参数
    static void init_inflow(const Config* config, int rank);

private:
    // 气体模型：标准大气 (对应 Fortran air 子程序)
    // h_km: 高度(km), t, p, rho, a: 输出引用
    static void calculate_standard_atmosphere(real_t h_km, real_t& t, real_t& p, real_t& rho, real_t& a);

    // 辅助函数：计算滞止参数 (对应 Fortran pr_stag)
    static void calculate_stagnation_properties();
};

} // namespace Hyres