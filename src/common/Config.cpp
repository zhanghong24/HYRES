#include "common/Config.h"
#include <fstream>
#include <iostream>
#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

namespace Hyres {

void Config::load(const std::string& filename, bool verbose) {
    if (verbose) {
        spdlog::info(">>> Reading configuration from: {}", filename);
    }

    std::ifstream f(filename);
    if (!f.is_open()) {
        spdlog::error("Failed to open config file: {}", filename);
        exit(EXIT_FAILURE);
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::parse_error& e) {
        spdlog::error("JSON Parse Error: {}", e.what());
        exit(EXIT_FAILURE);
    }

    // 1. Inflow
    auto& j_in = j["inflow"];
    inflow.mach     = j_in.value("mach", 6.0);
    inflow.alpha    = j_in.value("alpha", 0.0);
    inflow.sideslip = j_in.value("sideslip", 0.0);
    inflow.reynolds = j_in.value("reynolds", 0.0);
    inflow.T_wall   = j_in.value("T_wall", 300.0);
    inflow.height   = j_in.value("height", 0.0);
    inflow.T_ref    = j_in.value("T_ref", 288.15);
    inflow.P_ref    = j_in.value("P_ref", 101325.0);
    inflow.rho_ref  = j_in.value("rho_ref", 1.225);
    inflow.V_ref    = j_in.value("V_ref", 0.0);
    inflow.ndim     = j_in.value("ndim", 3);

    // 2. Force Ref
    auto& j_fr = j["force_ref"];
    forceRef.is_full_field = j_fr.value("is_full_field", 1);
    forceRef.area_ref      = j_fr.value("area_ref", 1.0);
    forceRef.len_ref_Re    = j_fr.value("len_ref_Re", 1.0);
    forceRef.len_ref_grid  = j_fr.value("len_ref_grid", 1.0);
    // 读取数组
    if(j_fr.contains("point_ref")) {
        forceRef.point_ref[0] = j_fr["point_ref"][0];
        forceRef.point_ref[1] = j_fr["point_ref"][1];
        forceRef.point_ref[2] = j_fr["point_ref"][2];
    }

    // 3. Filenames
    auto& j_file = j["filenames"];
    files.grid        = j_file.value("grid", "grid.x");
    files.bc          = j_file.value("bc", "grid.inp");
    files.output_dir  = j_file.value("output_dir", "result");

    // 4. Control
    auto& j_ctrl = j["control"];
    control.start_mode      = j_ctrl.value("start_mode", 0);
    control.max_steps       = j_ctrl.value("max_steps", 100);
    control.save_interval   = j_ctrl.value("save_interval", 100);
    control.force_interval  = j_ctrl.value("force_interval", 100);
    control.res_interval    = j_ctrl.value("residual_interval", 1);

    // 5. Time Step
    auto& j_ts = j["time_step"];
    timeStep.mode        = j_ts.value("mode", 0);
    timeStep.cfl         = j_ts.value("cfl", 1.0);
    timeStep.dt_nondim   = j_ts.value("dt_nondim", 0.01);
    timeStep.dt_rate     = j_ts.value("dt_growth_rate", 1.0);
    timeStep.dt_dts      = j_ts.value("dt_dts", 0.01);
    timeStep.max_substeps= j_ts.value("max_substeps", 1);
    timeStep.substep_tol = j_ts.value("substep_tol", 0.01);

    // 6. Physics
    auto& j_phy = j["physics"];
    physics.viscous_mode   = j_phy.value("viscous_mode", 1);
    physics.chemistry_mode = j_phy.value("chemistry_mode", 0);
    physics.thermo_model   = j_phy.value("thermo_model", 1);

    // 7. Scheme
    auto& j_sch = j["scheme"];
    scheme.lhs_method = j_sch.value("lhs_method", 2);
    scheme.scheme_id  = j_sch.value("scheme_id", 4);
    scheme.flux_id    = j_sch.value("flux_id", 3);
    scheme.limiter_id = j_sch.value("limiter_id", 1);
    scheme.entropy_fix= j_sch.value("entropy_fix", 0.1);
    scheme.visc_spectral_radius = j_sch.value("visc_spectral_radius", 1.0);
    scheme.res_smoothing = j_sch.value("residual_smoothing", 0);

    // 8. Interpolation
    auto& j_int = j["interpolation"];
    interp.xk = j_int.value("xk", -1.0);
    interp.xb = j_int.value("xb", 1.0);
    interp.visc_k2 = j_int.value("visc_k2", 0.0);
    interp.visc_k4 = j_int.value("visc_k4", 0.0);

    // 9. Chemistry
    auto& j_chem = j["chemistry"];
    chemistry.gas_model     = j_chem.value("gas_model", "air");
    chemistry.source_method = j_chem.value("source_method", 0);
    chemistry.rad_model     = j_chem.value("radiation_model", 0);

    // 10. Boundary Features (GCL_CIC)
    auto& j_bf = j["boundary_features"];
    boundary.gcl = j_bf.value("gcl", 0);
    if (j_bf.contains("cic")) {
        for(int i=0; i<5; ++i) boundary.cic[i] = j_bf["cic"][i];
    }

    // 11. Connectivity
    auto& j_conn = j["connectivity"];
    connect.read_mode = j_conn.value("read_mode", 0);
    connect.file      = j_conn.value("file", "conect.dat");
    connect.order     = j_conn.value("order", 0);
    connect.dist_tol  = j_conn.value("dist_tol", 0.0);

    if (verbose) {
        spdlog::info(">>> Configuration loaded successfully.");
    }
}

void Config::log_config() const {
    spdlog::info("---------------------------------------------");
    spdlog::info(" [Control] Max Steps: {}, Save Interval: {}", control.max_steps, control.save_interval);
    spdlog::info(" [Inflow]  Mach: {}, Reynolds: {}", inflow.mach, inflow.reynolds);
    spdlog::info(" [Scheme]  ID: {}, Flux: {}, LHS: {}", scheme.scheme_id, scheme.flux_id, scheme.lhs_method);
    spdlog::info(" [Files]   Grid: {}", files.grid);
    spdlog::info("---------------------------------------------");
}

} // namespace Hyres