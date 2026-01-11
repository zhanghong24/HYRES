#pragma once

#include <vector>
#include <array>
#include <cmath>
#include "common/Types.h"

namespace Hyres {

struct PointIndex {
    int block_id; // 全局 Block ID
    int i, j, k;  // 0-based
    int global_buffer_index; // 在全局 dq_npp 数组中的下标
};

// 一个奇异点组（物理上重合的一组点）
struct SingularityGroup {
    std::vector<PointIndex> points;
};

/**
 * @brief 全局物理参数容器 (对应 Fortran global_variables)
 * 使用单例模式访问，存储物理常数、来流条件、参考状态等。
 */
struct GlobalData {
    // 获取单例实例
    static GlobalData& getInstance() {
        static GlobalData instance;
        return instance;
    }

    // ========================================================
    // 0. 网格拓扑信息 (Grid Topology)
    // ========================================================
    int n_blocks = 0;

    // ========================================================
    // 1. 基础物理常数 (Physical Constants)
    // ========================================================
    real_t pi = 3.14159265358979323846;
    real_t gamma = 1.4;        // 比热比 (gama)
    real_t gas_constant = 287.0; // 气体常数 R (rjmk, 待初始化)
    real_t pr_laminar = 0.72;  // 层流普朗特数 (prl)
    real_t pr_turbulent = 0.90;// 湍流普朗特数 (prt)

    // ========================================================
    // 2. 输入参数 (Config / User Input)
    // ========================================================
    real_t alpha_deg = 0.0;    // 攻角 (角度)
    real_t beta_deg = 0.0;     // 侧滑角 (角度)
    real_t alpha_rad = 0.0;    // 攻角 (弧度, attack)
    real_t beta_rad = 0.0;     // 侧滑角 (弧度, sideslip)

    real_t mach = 0.0;         // 马赫数 (moo)
    real_t reynolds = 0.0;     // 雷诺数 (reynolds)
    real_t height = 0.0;       // 高度 (height)

    // 参考状态 (Dimensional Reference Values from Config)
    real_t t_ref = 0.0;        // 参考温度 (tref)
    real_t p_ref = 0.0;        // 参考压力 (pref)
    real_t rho_ref = 0.0;      // 参考密度 (rref)
    
    // 特征长度 (Characteristic Lengths)
    real_t len_ref_Re = 1.0;   // 雷诺数特征长度 (lfref)
    real_t len_ref_grid = 1.0; // 网格特征长度 (lref)

    // ========================================================
    // 3. 计算出的无量纲来流状态 (Nondimensional Inflow State)
    // ========================================================
    // 速度分量
    real_t u_inf = 0.0;        // uoo
    real_t v_inf = 0.0;        // voo
    real_t w_inf = 0.0;        // woo

    // 热力学状态 (无量纲)
    real_t rho_inf = 1.0;      // roo (通常设为 1.0)
    real_t p_inf = 0.0;        // poo
    real_t t_inf = 1.0;        // too (通常设为 1.0)
    real_t e_inf = 0.0;        // eoo (单位体积总能)
    real_t h_inf = 0.0;        // hoo (总焓)
    real_t c_inf = 0.0;        // coo (无量纲声速)

    // 守恒变量初值 Q_inf (Conservative Variables)
    // 对应 q1oo, q2oo, q3oo, q4oo, q5oo
    std::array<real_t, 5> q_inf; 

    // ========================================================
    // 4. 有量纲导出参数 (Dimensional Derived Values)
    // ========================================================
    real_t mu_ref = 0.0;       // 参考粘性系数 (visloo)
    real_t c_ref_dim = 0.0;    // 有量纲声速 (ccoo)
    real_t v_ref_dim = 0.0;    // 有量纲参考速度 (vref)
    real_t rvl = 0.0;          // rvl = rho_ref * v_ref / l_ref (中间变量)
    real_t beta1 = 0.0;        // beta1 (中间变量)

    // ========================================================
    // 5. 化学组分与气体模型 (Chemistry & Gas Model)
    // ========================================================
    int n_species = 1;         // 组分数 (ns)
    
    // 分子量相关
    real_t mol_weight_avg = 28.96;     // 平均分子量 (mref)
    real_t mol_weight_avg_inv = 0.0;   // 平均分子量倒数 (mref1)

    // 组分数组 (Chemistry Arrays)
    std::vector<real_t> mol_weights;       // 分子量 (ws)
    std::vector<real_t> mol_weights_inv;   // 分子量倒数 (ws1)
    std::vector<real_t> mass_fractions_init; // 初始质量分数 (cn_init)
    
    // 辅助: ms, ms1 (用于化学反应源项?)
    std::vector<real_t> ms;
    std::vector<real_t> ms1;

    // ========================================================
    // 6. 辅助系数 (Auxiliary Coefficients)
    // ========================================================
    real_t cq_lam = 0.0;       // 层流热通量系数
    real_t cq_tur = 0.0;       // 湍流热通量系数
    real_t sutherland_c = 0.0; // Sutherland常数/Tref (visc)
    real_t re_inv = 0.0;       // 1.0 / Reynolds (re)

    // 滞止参数 (Stagnation Properties)
    real_t p_stag = 0.0;       // 总压 (p0)
    real_t t_stag = 0.0;       // 总温 (t0)
    real_t rho_stag = 0.0;     // 总密度 (r0)

    real_t timedt;

    // ========================================================
    // 【新增】多点连接 (Singularity / Multi-point Constraints)
    // ========================================================
    
    // 1. 全局奇异点组列表 (用于 boundary_match_dq_3pp 计算平均值)
    // 对应 Fortran: cdate_ex (connection date extended)
    std::vector<SingularityGroup> singularity_groups;

    // 2. MPI 通信辅助数组
    // 对应 Fortran: nppos_local (每个rank的点数), ipp_st_local (偏移)
    std::vector<int> nppos_local; 
    std::vector<int> ipp_st_local;
    int total_npp = 0; // 全局奇异点总数

    // 3. 全局通信 Buffer (MPI_Allgatherv 的结果)
    // 对应 Fortran: dq_npp, pv_npp
    std::vector<real_t> dq_npp_global; // Size: total_npp * 5
    std::vector<real_t> pv_npp_global; // Size: total_npp * 6
    
    // 4. 本地发送 Buffer (Gather 后的结果)
    // 对应 Fortran: dq_npp_local, pv_npp_local
    std::vector<real_t> dq_npp_local;
    std::vector<real_t> pv_npp_local;

private:
    GlobalData() = default; // 私有构造，强制单例
    GlobalData(const GlobalData&) = delete;
    GlobalData& operator=(const GlobalData&) = delete;
};

} // namespace Hyres