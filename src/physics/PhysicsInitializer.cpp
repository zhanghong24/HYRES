#include "physics/PhysicsInitializer.h"
#include "spdlog/spdlog.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

namespace Hyres {

void PhysicsInitializer::init_inflow(const Config* config, int rank) {
    GlobalData& g = GlobalData::getInstance();

    // ========================================================
    // 1. 基础物理常数设置 (复刻 Fortran 硬编码值)
    // ========================================================
    g.pi = 3.14159265358979323846;
    real_t r_univ = 8.31434; // Fortran: rjmk
    
    // 气体常数稍后根据组分计算，但 air 子程序中使用了 287.053
    // 这里先给一个初始值，避免未初始化
    g.gas_constant = 287.053; 
    
    // ========================================================
    // 2. 读取输入参数
    // ========================================================
    g.alpha_deg = config->inflow.alpha;
    g.beta_deg = config->inflow.sideslip;
    g.mach = config->inflow.mach;
    g.reynolds = config->inflow.reynolds; // 初值，后续会修正
    g.height = config->inflow.height;
    
    g.t_ref = config->inflow.T_ref;
    g.p_ref = config->inflow.P_ref;
    g.rho_ref = config->inflow.rho_ref;
    
    // 【核心修正】使用用户指定的变量名 forceRef
    g.len_ref_Re = config->forceRef.len_ref_Re;     // lfref
    g.len_ref_grid = config->forceRef.len_ref_grid; // lref

    // ========================================================
    // 3. 几何角度与来流方向
    // ========================================================
    g.alpha_rad = g.alpha_deg * g.pi / 180.0;
    g.beta_rad = g.beta_deg * g.pi / 180.0;

    g.u_inf = std::cos(g.alpha_rad) * std::cos(g.beta_rad);
    g.v_inf = std::sin(g.alpha_rad) * std::cos(g.beta_rad);
    g.w_inf = std::sin(g.beta_rad);

    // ========================================================
    // 4. 无量纲基准
    // ========================================================
    g.rho_inf = 1.0; // roo
    g.t_inf = 1.0;   // too

    // ========================================================
    // 5. 雷诺数修正
    // ========================================================
    // reynolds = (reynolds/lfref)*lref
    g.reynolds = (g.reynolds / g.len_ref_Re) * g.len_ref_grid;

    // ========================================================
    // 6. 化学组分初始化 (复刻 read_perfect_gasmodel)
    // ========================================================
    std::string gas_model = config->chemistry.gas_model;
    int ns = 0;

    if (gas_model == "air") {
        ns = 2;
        g.n_species = ns;
        
        // 分配内存
        g.mol_weights.resize(ns);         // ws
        g.mol_weights_inv.resize(ns);     // ws1
        g.ms.resize(ns);
        g.ms1.resize(ns);
        g.mass_fractions_init.resize(ns); // cn_init

        // N2
        g.mol_weights[0] = 28.0; 
        g.mass_fractions_init[0] = 0.79;

        // O2
        g.mol_weights[1] = 32.0; 
        g.mass_fractions_init[1] = 0.21;

        // 设置全局参数
        g.gamma = 1.40;
        g.pr_laminar = 0.72; // prl
        g.pr_turbulent = 0.90; // prt

    } else {
        if (rank == 0) {
            spdlog::error("Unknown gas_model: {}. Only 'air' is supported.", gas_model);
        }
        exit(EXIT_FAILURE); 
    }

    // --- 计算混合气体属性 (对应 Fortran init_inflow 循环) ---
    
    // 3.1 分子量单位转换 (g/mol -> kg/mol)
    for(int i=0; i<ns; ++i) {
        g.mol_weights[i] *= 1.0e-3;
    }

    // 3.2 计算分子量倒数
    for(int i=0; i<ns; ++i) {
        g.mol_weights_inv[i] = 1.0 / g.mol_weights[i];
    }

    // 3.3 计算平均分子量倒数 (mref1)
    g.mol_weight_avg_inv = 0.0;
    for(int i=0; i<ns; ++i) {
        g.mol_weight_avg_inv += g.mass_fractions_init[i] * g.mol_weights_inv[i];
    }

    // 3.4 计算平均分子量 (mref)
    g.mol_weight_avg = 1.0 / g.mol_weight_avg_inv;

    // 3.5 计算组分参数 ms, ms1
    for(int i=0; i<ns; ++i) {
        g.ms[i] = g.mol_weights[i] * g.mol_weight_avg_inv;
        g.ms1[i] = 1.0 / g.ms[i];
    }

    // 【关键】计算混合气体常数 R_spec
    // Fortran 逻辑隐含：R_spec = rjmk * mref1 (Universal Gas Constant / M_avg)
    real_t r_spec_mix = r_univ * g.mol_weight_avg_inv;
    g.gas_constant = r_spec_mix; // 更新全局气体常数 (约为 289.1)

    // ========================================================
    // 7. 大气环境初始化 (Reference State)
    // ========================================================
    real_t ccon = 0.0; // Sutherland Constant Term

    if (g.height < 0.0000) {
        // --- 用户自定义分支 ---
        // 逻辑：利用 P = rho * R * T 补全缺失量
        // Fortran: rref = pref/(tref*rjmk*mref1) -> rref = pref/(tref * r_spec_mix)

        if (g.p_ref > 0 && g.t_ref > 0 && g.rho_ref <= 0) {
            g.rho_ref = g.p_ref / (g.t_ref * r_spec_mix);
        } else if (g.rho_ref > 0 && g.t_ref > 0 && g.p_ref <= 0) {
            g.p_ref = g.rho_ref * g.t_ref * r_spec_mix;
        } else if (g.p_ref > 0 && g.rho_ref > 0 && g.t_ref <= 0) {
            g.t_ref = g.p_ref / (g.rho_ref * r_spec_mix);
        }

        // Sutherland Viscosity
        // ccon = (1 + 110.4/273.15) / (1 + 110.4/tref)
        ccon = (1.0 + 110.4/273.15) / (1.0 + 110.4/g.t_ref);
        g.mu_ref = std::sqrt(g.t_ref/273.15) * ccon * 1.715e-5;

        // Sound Speed
        // ccoo = sqrt(gama * pref / rref)
        g.c_ref_dim = std::sqrt(g.gamma * g.p_ref / g.rho_ref);

    } else {
        // --- 标准大气分支 ---
        // 调用 air 子程序 (使用标准 R=287.053 计算 T, P)
        calculate_standard_atmosphere(g.height, g.t_ref, g.p_ref, g.rho_ref, g.c_ref_dim);
        
        // 【关键】Fortran 在调用 air 后会强制重算 rho_ref
        // rref = pref/(tref*rjmk*mref1)
        // 即使 air 内部用的是 R=287.053，这里也会用 R_mix (289.1) 修正 rho
        g.rho_ref = g.p_ref / (g.t_ref * r_spec_mix);

        // Sutherland Viscosity
        ccon = (1.0 + 110.4/273.15) / (1.0 + 110.4/g.t_ref);
        g.mu_ref = std::sqrt(g.t_ref/273.15) * ccon * 1.715e-5;
    }

    // 8. 输出参考状态 (仅 Master 输出)
    if (rank == 0) {
        spdlog::info("Inflow Conditions:");
        spdlog::info("  Height:   {:.4e}", g.height);
        spdlog::info("  Temp:     {:.4e}", g.t_ref);
        spdlog::info("  Pressure: {:.4e}", g.p_ref);
        spdlog::info("  Density:  {:.4e}", g.rho_ref);
        spdlog::info("  Sound Spd:{:.4e}", g.c_ref_dim);
    }

    // ========================================================
    // 9. 导出无量纲参数
    // ========================================================
    // vref = ccoo * moo
    g.v_ref_dim = g.c_ref_dim * g.mach;

    // rvl = rref * vref / lref
    g.rvl = g.rho_ref * g.v_ref_dim / g.len_ref_grid;
    
    // beta1 = rjmk*tref/(vref*vref*mref) -> R_spec * T / V^2
    g.beta1 = r_spec_mix * g.t_ref / (g.v_ref_dim * g.v_ref_dim);

    // Reynolds recalculation if height >= 0
    if (g.height >= 0.0) {
        g.reynolds = g.rho_ref * g.v_ref_dim * g.len_ref_grid / g.mu_ref;
    }

    // poo = 1.0/(gama*moo*moo)
    g.p_inf = 1.0 / (g.gamma * g.mach * g.mach);
    
    // coo = sqrt(gama*poo/roo)
    g.c_inf = std::sqrt(g.gamma * g.p_inf / g.rho_inf); // roo=1.0

    // eoo = poo/(gama-1) + 0.5*roo*(vel^2)
    g.e_inf = g.p_inf / (g.gamma - 1.0) + 0.5 * g.rho_inf * (1.0);

    // hoo = (eoo + poo)/roo
    g.h_inf = (g.e_inf + g.p_inf) / g.rho_inf;

    // 填充 Q_inf
    g.q_inf[0] = g.rho_inf;
    g.q_inf[1] = g.rho_inf * g.u_inf;
    g.q_inf[2] = g.rho_inf * g.v_inf;
    g.q_inf[3] = g.rho_inf * g.w_inf;
    g.q_inf[4] = g.e_inf;

    // Log Derived
    if (rank == 0) {
        spdlog::info("  Reynolds: {:.4e}", g.reynolds);
        spdlog::info("  Ref Visc: {:.4e}", g.mu_ref);
        spdlog::info("  Mach:     {:.4e}", g.mach);
        spdlog::info("  V Ref:    {:.4e}", g.v_ref_dim);
    }

    // 10. 滞止参数
    calculate_stagnation_properties();

    // 11. 辅助系数
    g.cq_lam = 1.0 / ((g.gamma - 1.0) * g.mach * g.mach * g.pr_laminar);
    g.cq_tur = 1.0 / ((g.gamma - 1.0) * g.mach * g.mach * g.pr_turbulent);
    g.sutherland_c = 110.4 / g.t_ref;
    g.re_inv = 1.0 / g.reynolds;
}

// =========================================================================
// 子程序：calculate_standard_atmosphere (复刻 Fortran air)
// =========================================================================
void PhysicsInitializer::calculate_standard_atmosphere(real_t h1, real_t& t, real_t& p, real_t& rho, real_t& a) {
    real_t h = h1 * 1000.0;
    
    // Standard Atmosphere Constants (Hardcoded in Fortran)
    real_t r = 287.053; 
    real_t g0 = 9.80665;
    real_t rp = 6.37111e6;
    real_t g = std::pow(rp / (rp + h), 2) * g0;

    real_t t0 = 288.15;      real_t p0 = 10.1325e2;     real_t rho0 = 1.225;
    real_t t11 = 216.65;     real_t p11 = 2.2632e2;     real_t rho11 = 3.6392e-1;
    real_t t20 = t11;        real_t p20 = 5.4747e1;     real_t rho20 = 8.8035e-2;
    real_t t32 = 228.65;     real_t p32 = 8.6789;       real_t rho32 = 1.3225e-2;
    real_t t47 = 270.65;     real_t p47 = 1.1090;       real_t rho47 = 1.4275e-3;
    real_t t52 = t47;        real_t p52 = 5.8997e-1;    real_t rho52 = 7.5943e-4;
    real_t t61 = 252.65;     real_t p61 = 1.8209e-1;    real_t rho61 = 2.5109e-4;
    real_t t79 = 180.65;     real_t p79 = 1.0376e-2;    real_t rho79 = 2.0010e-5;
    real_t t90 = t79;        real_t p90 = 1.6437e-3;    real_t rho90 = 3.4165e-6;
    real_t t100 = 210.02;    real_t p100 = 3.0070e-4;   real_t rho100 = 5.6044e-7;
    real_t t110 = 257.00;    real_t p110 = 7.3527e-5;   real_t rho110 = 9.7081e-8;
    real_t t120 = 349.49;    real_t p120 = 2.5209e-5;   real_t rho120 = 2.2222e-8;
    real_t t150 = 892.79;    real_t p150 = 5.0599e-6;   real_t rho150 = 2.0752e-9;
    real_t t160 = 1022.20;   real_t p160 = 3.6929e-6;   real_t rho160 = 1.2336e-9;
    real_t t170 = 1103.40;   real_t p170 = 2.7915e-6;   real_t rho170 = 7.8155e-10;
    real_t t190 = 1205.40;   real_t p190 = 1.6845e-6;   real_t rho190 = 3.5807e-10;
    real_t t230 = 1322.30;   real_t p230 = 6.7138e-7;   real_t rho230 = 1.5640e-10;
    real_t t300 = 1432.10;   real_t p300 = 1.8828e-7;   real_t rho300 = 1.9159e-11;
    real_t t400 = 1487.40;   real_t p400 = 4.0278e-8;   real_t rho400 = 2.8028e-12;
    real_t t500 = 1499.20;   real_t p500 = 1.0949e-8;   real_t rho500 = 5.2148e-13;
    real_t t600 = 1506.10;   real_t p600 = 3.4475e-9;   real_t rho600 = 1.1367e-13;
    real_t t700 = 1507.60;   real_t p700 = 1.1908e-9;   real_t rho700 = 1.5270e-13;

    if (h <= 11019.0) {
        real_t al1 = (t11 - t0) / 11019.0;
        t = t0 + al1 * h;
        p = p0 * std::pow(t / t0, -g / (r * al1));
        rho = rho0 * std::pow(t / t0, -1.0 - g / (r * al1));
    } else if (h <= 20063.0) {
        t = t11;
        p = p11 * std::exp(-g * (h - 11019.0) / (r * t11));
        rho = rho11 * std::exp(-g * (h - 11019.0) / (r * t11));
    } else if (h <= 32162.0) {
        real_t al2 = (t32 - t20) / (32162.0 - 20063.0);
        t = t11 + al2 * (h - 20063.0);
        p = p20 * std::pow(t / t11, -g / (r * al2));
        rho = rho20 * std::pow(t / t11, -1.0 - g / (r * al2));
    } else if (h <= 47350.0) {
        real_t al3 = (t47 - t32) / (47350.0 - 32162.0);
        t = t32 + al3 * (h - 32162.0);
        p = p32 * std::pow(t / t32, -g / (r * al3));
        rho = rho32 * std::pow(t / t32, -1.0 - g / (r * al3));
    } else if (h <= 52429.0) {
        t = t47;
        p = p47 * std::exp(-g * (h - 47350.0) / (r * t47));
        rho = rho47 * std::exp(-g * (h - 47350.0) / (r * t47));
    } else if (h <= 61591.0) {
        real_t al4 = (t61 - t52) / (61591.0 - 52429.0);
        t = t47 + al4 * (h - 52429.0);
        p = p52 * std::pow(t / t47, -g / (r * al4));
        rho = rho52 * std::pow(t / t47, -1.0 - g / (r * al4));
    } else if (h <= 79994.0) {
        real_t al5 = (t79 - t61) / (79994.0 - 61591.0);
        t = t61 + al5 * (h - 61591.0);
        p = p61 * std::pow(t / t61, -g / (r * al5));
        rho = rho61 * std::pow(t / t61, -1.0 - g / (r * al5));
    } else if (h <= 90000.0) {
        t = t79;
        p = p79 * std::exp(-g * (h - 79994.0) / (r * t79));
        rho = rho79 * std::exp(-g * (h - 79994.0) / (r * t79));
    } else if (h <= 100000.0) {
        real_t al6 = (t100 - t90) / 10000.0;
        t = t79 + al6 * (h - 90000.0);
        p = p90 * std::pow(t / t79, -g / (r * al6));
        rho = rho90 * std::pow(t / t79, -1.0 - g / (r * al6));
    } else if (h <= 110000.0) {
        real_t al7 = (t110 - t100) / 10000.0;
        t = t100 + al7 * (h - 100000.0);
        p = p100 * std::pow(t / t100, -g / (r * al7));
        rho = rho100 * std::pow(t / t100, -1.0 - g / (r * al7));
    } else if (h <= 120000.0) {
        real_t al8 = (t120 - t110) / 10000.0;
        t = t110 + al8 * (h - 110000.0);
        p = p110 * std::pow(t / t110, -g / (r * al8));
        rho = rho110 * std::pow(t / t110, -1.0 - g / (r * al8));
    } else if (h <= 150000.0) {
        real_t al9 = (t150 - t120) / 30000.0;
        t = t120 + al9 * (h - 120000.0);
        p = p120 * std::pow(t / t120, -g / (r * al9));
        rho = rho120 * std::pow(t / t120, -1.0 - g / (r * al9));
    } else if (h <= 160000.0) {
        real_t al10 = (t160 - t150) / 10000.0;
        t = t150 + al10 * (h - 150000.0);
        p = p150 * std::pow(t / t150, -g / (r * al10));
        rho = rho150 * std::pow(t / t150, -1.0 - g / (r * al10));
    } else if (h <= 170000.0) {
        real_t al11 = (t170 - t160) / 10000.0;
        t = t160 + al11 * (h - 160000.0);
        p = p160 * std::pow(t / t160, -g / (r * al11));
        rho = rho160 * std::pow(t / t160, -1.0 - g / (r * al11));
    } else if (h <= 190000.0) {
        real_t al12 = (t190 - t170) / 20000.0;
        t = t170 + al12 * (h - 170000.0);
        p = p170 * std::pow(t / t170, -g / (r * al12));
        rho = rho170 * std::pow(t / t170, -1.0 - g / (r * al12));
    } else if (h <= 230000.0) {
        real_t al13 = (t230 - t190) / 40000.0;
        t = t190 + al13 * (h - 190000.0);
        p = p190 * std::pow(t / t190, -g / (r * al13));
        rho = rho190 * std::pow(t / t190, -1.0 - g / (r * al13));
    } else if (h <= 300000.0) {
        real_t al14 = (t300 - t230) / 70000.0;
        t = t230 + al14 * (h - 230000.0);
        p = p230 * std::pow(t / t230, -g / (r * al14));
        rho = rho230 * std::pow(t / t230, -1.0 - g / (r * al14));
    } else if (h <= 400000.0) {
        real_t al15 = (t400 - t300) / 100000.0;
        t = t300 + al15 * (h - 300000.0);
        p = p300 * std::pow(t / t300, -g / (r * al15));
        rho = rho300 * std::pow(t / t300, -1.0 - g / (r * al15));
    } else if (h <= 500000.0) {
        real_t al16 = (t500 - t400) / 100000.0;
        t = t400 + al16 * (h - 400000.0);
        p = p400 * std::pow(t / t400, -g / (r * al16));
        rho = rho400 * std::pow(t / t400, -1.0 - g / (r * al16));
    } else if (h <= 600000.0) {
        real_t al17 = (t600 - t500) / 100000.0;
        t = t500 + al17 * (h - 500000.0);
        p = p500 * std::pow(t / t500, -g / (r * al17));
        rho = rho500 * std::pow(t / t500, -1.0 - g / (r * al17));
    } else if (h <= 700000.0) {
        real_t al18 = (t700 - t600) / 100000.0;
        t = t600 + al18 * (h - 600000.0);
        p = p600 * std::pow(t / t600, -g / (r * al18));
        rho = rho600 * std::pow(t / t600, -1.0 - g / (r * al18));
    }

    a = std::sqrt(1.4 * r * t);
    p = p * 100.0; // mbar -> Pa
}

// =========================================================================
// 子程序：calculate_stagnation_properties (复刻 pr_stag)
// =========================================================================
void PhysicsInitializer::calculate_stagnation_properties() {
    GlobalData& g = GlobalData::getInstance();
    
    // Formula: p0 = p * (1 + (g-1)/2 * M^2)^(g/(g-1))
    real_t factor = 1.0 + 0.5 * (g.gamma - 1.0) * g.mach * g.mach;
    
    g.t_stag = g.t_inf * factor;
    g.p_stag = g.p_inf * std::pow(factor, g.gamma / (g.gamma - 1.0));
    g.rho_stag = g.rho_inf * std::pow(factor, 1.0 / (g.gamma - 1.0));
}

} // namespace Hyres