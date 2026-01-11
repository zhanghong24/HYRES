#pragma once
#include <string>
#include <vector>
#include <array>
#include "common/Types.h"

namespace Hyres {

struct InflowConfig {
    real_t mach;        // moo
    real_t alpha;       // attack
    real_t sideslip;    // sideslip
    real_t reynolds;    // reynolds
    real_t T_wall;      // twall
    real_t height;      // height
    real_t T_ref;       // tref
    real_t P_ref;       // pref
    real_t rho_ref;     // rref
    real_t V_ref;       // vref
    int ndim;           // ndim
};

struct ForceRefConfig {
    int is_full_field;  // nwholefield
    real_t area_ref;    // sref
    real_t len_ref_Re;  // lfref
    real_t len_ref_grid;// lref
    std::array<real_t, 3> point_ref; // xref, yref, zref
};

struct FileConfig {
    std::string grid;   // gridname
    std::string bc;     // bcname
    std::string output_dir;  // output dir
};

struct ControlConfig {
    int start_mode;     // nstart
    int max_steps;      // nomax
    int save_interval;  // ndisk
    int force_interval; // nforce
    int res_interval;   // nwerror
};

struct TimeStepConfig {
    int mode;           // ntmst
    real_t cfl;         // cfl
    real_t dt_nondim;   // timedt
    real_t dt_rate;     // timedt_rate
    real_t dt_dts;      // dtdts
    int max_substeps;   // nsubstmx
    real_t substep_tol; // tolsub
};

struct PhysicsConfig {
    int viscous_mode;   // nvis (0=Euler, 1=NS)
    int chemistry_mode; // nchem
    int thermo_model;   // ntmodel
};

struct SchemeConfig {
    int lhs_method;     // nlhs
    int scheme_id;      // nscheme
    int flux_id;        // nflux
    int limiter_id;     // nlimiter
    real_t entropy_fix; // efix
    real_t visc_spectral_radius; // csrv
    int res_smoothing;  // nsmooth
};

struct InterpConfig {
    real_t xk;
    real_t xb;
    real_t visc_k2;     // c_k2
    real_t visc_k4;     // c_k4
};

struct ChemConfig {
    std::string gas_model; // gasmodel
    int source_method;    // nchem_source
    int rad_model;        // nchem_rad
};

struct BoundaryFeatConfig {
    int gcl;              // GCL
    std::array<int, 5> cic; // CIC1 - CIC5
};

struct ConnectConfig {
    int read_mode;        // connect_point
    std::string file;     // conpointname
    int order;            // connect_order
    real_t dist_tol;      // dis_tao
};

// 主配置类
class Config {
public:
    InflowConfig inflow;
    ForceRefConfig forceRef;
    FileConfig files;
    ControlConfig control;
    TimeStepConfig timeStep;
    PhysicsConfig physics;
    SchemeConfig scheme;
    InterpConfig interp;
    ChemConfig chemistry;
    BoundaryFeatConfig boundary;
    ConnectConfig connect;

    void load(const std::string& filename, bool verbose = false);
    void log_config() const;
};

} // namespace Hyres