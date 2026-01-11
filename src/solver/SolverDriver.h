#pragma once

#include <vector>
#include <memory>
#include "data/Block.h"
#include "common/Config.h"
#include "common/MpiContext.h"
#include "solver/BoundaryManager.h"
#include <iomanip> // 用于 std::scientific, std::setprecision
#include <algorithm> // 用于 std::min, std::max

namespace Hyres {

class SolverDriver {
public:
    // 构造函数：接收初始化好的 Block 列表和配置
    SolverDriver(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi);
    
    // 析构函数
    ~SolverDriver();

    // 核心入口：开始主循环
    void solve();

private:
    // 单步时间推进 (Time Marching Step) - 对应 Fortran 的 time_integration
    void run_time_step(int step); 

    // 检查是否保存文件
    void check_io_output(int step);

    // 检查残差收敛性
    void check_residual(int step);

    void exchange_bc();

    void boundary_all();

    void update_derived_variables();

    void update_sound_speed(Block* b);

    void update_conservative_vars(Block* b);

    void reset_residuals(Block* b);

    void update_thermodynamics(Block* b);

    void update_laminar_viscosity(Block* b);

    void calculate_timestep();

    void spectrum_tgh(Block* b);

    void set_spectvis_to_0(Block* b);

    void spectinv(Block* b);

    void spectvisl(Block* b);

    void localdt0(Block* b);

    void calculate_viscous_rhs();

    void compute_block_viscous_rhs(Block* b);

    void uvwt_der_4th_virtual(Block* b, HyresArray<real_t>& duvwt);

    void compute_duvwt_node_line_virtual(
        int nmax, int ni,
        const std::vector<std::array<real_t, 4>>& uvwt, 
        std::vector<std::array<real_t, 4>>& duvwt,
        int offset
    );

    void uvwt_der_4th_half_virtual(Block* b, HyresArray<real_t>& duvwt_mid);

    void compute_duvwt_half_line_virtual(
        int nmax, int ni,
        const std::vector<std::array<real_t, 4>>& uvwt, 
        std::vector<std::array<real_t, 4>>& duvwt,
        int offset
    );

    void compute_value_line_half_virtual(int nmax, int ni, 
                                         const std::vector<std::vector<real_t>>& q, 
                                         std::vector<std::vector<real_t>>& q_half, 
                                         int offset);

    void compute_value_half_node_ijk(
        int nmax, int ni, int n_comps,
        int ip, int jp, int kp, // 掩码标志
        const std::vector<std::vector<real_t>>& q, // Input: [comp][index]
        std::vector<std::vector<real_t>>& q_half   // Output: [comp][index]
    );

    void compute_value_half_node(
        int nmax, int ni, int n_comps,
        const std::vector<std::vector<real_t>>& q, 
        std::vector<std::vector<real_t>>& q_half
    );

    void compute_duvwt_dxyz(
        int nmax, int ni,
        const std::vector<std::vector<real_t>>& duvwt,
        const std::vector<std::vector<real_t>>& kxyz,
        const std::vector<real_t>& vol,
        std::vector<std::vector<real_t>>& duvwtdxyz
    );

    void compute_flux_vis_line_new(int nmax, int ni,
                                   const std::vector<std::vector<real_t>>& uvwt,
                                   const std::vector<std::vector<real_t>>& duvwtdxyz,
                                   const std::vector<std::vector<real_t>>& kxyz,
                                   int idx_nx, int idx_ny, int idx_nz,
                                   const std::vector<real_t>& vslt1,
                                   const std::vector<real_t>& vslt2,
                                   std::vector<std::vector<real_t>>& fv);

    void compute_flux_dxyz(
        int nmax, int ni, int nl,
        const std::vector<std::vector<real_t>>& f,  // Input Flux [nl][0:nmax]
        std::vector<std::vector<real_t>>& df        // Output Deriv [nl][nmax]
    );

    void compute_value_half_node_scalar(int nmax, int ni,
                                        const std::vector<real_t>& q, 
                                        std::vector<real_t>& q_half);

    void communicate_dq_npp();

    void exchange_bc_dq_vol();

    void boundary_match_dq_2pm();    

    void boundary_match_dq_3pp();
    
    void calculate_inviscous_rhs();

    void compute_block_inviscous_rhs(Block* b);

    // =========================================================================
    // 核心计算：Roe 通量格式 (目前为空)
    // 对应 Fortran: subroutine flux_Roe
    // =========================================================================
    void flux_roe_kernel(
        const real_t* ql_prim, const real_t* qr_prim, // 输入：左右原始变量 [rho, u, v, w, p]
        real_t nx, real_t ny, real_t nz,              // 输入：界面法向量 (包含面积)
        real_t gamma, real_t entropy_fix,             // 输入：气体参数与熵修正系数
        real_t* f_interface                           // 输出：界面通量 [5]
    );

    // =========================================================================
    // 驱动函数：计算一条线上的通量导数 (目前为空)
    // 对应 Fortran: flux_line (WCNS + Roe 分支)
    // =========================================================================
    void flux_line_wcns_roe(
        int ni, // 物理网格点数 (从 0 到 ni-1)
        const std::vector<std::array<real_t, 6>>& q_line_prim, // 输入：原始变量线 (带 Ghost)
        const std::vector<std::array<real_t, 6>>& q_line_cons, // 输入：守恒变量线 (备用)
        const std::vector<std::array<real_t, 5>>& trxyz,       // 输入：几何度量 (物理区)
        real_t efix,                                           // 输入：熵修正系数
        std::vector<std::array<real_t, 5>>& fc                 // 输出：通量差分/残差增量
    );

    void time_step_lhs_tgh();

    void set_boundary_dq_zero(Block* b);

    void solve_gauss_seidel_single(Block* b);

    void update_conservatives(Block* b);

    void store_rhs_nb(Block* b, std::vector<real_t>& rhs_nb);
    void lusgs_l(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta, real_t extra_param);
    void lusgs_u(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta, real_t extra_param);
    void gs_pr_l(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta);
    void gs_pr_u(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta);
    void matrix_vector_product_std(
        const real_t* prim, const real_t* metrics, const real_t* dq, 
        real_t* f_out, real_t rad, int npn, real_t gamma);

    void output_solution(int step_idx);
    void write_block_vts(Block* b, const std::string& filename);
    void write_global_vtm(const std::string& filename, const std::string& step_str, int total_blocks);

    void calculate_aerodynamic_forces(int step_idx);
    bool is_wall_boundary(int type);
    
    void accumulate_face_force(
        double px1, double py1, double pz1,
        double px2, double py2, double pz2,
        double px3, double py3, double pz3,
        double px4, double py4, double pz4,
        double x1, double y1, double z1,
        double x2, double y2, double z2,
        double x3, double y3, double z3,
        double x4, double y4, double z4,
        double xref, double yref, double zref,
        double* forces
    );

    void communicate_pv_npp();

    void exchange_bc_pv_vol();    
    void boundary_match_pv_2pm();
    void boundary_match_pv_3pp();
    void output_solution_plt(int step_idx);
    struct BlockMeta {
        int id;
        int ni, nj, nk;
        int owner_rank;
    };
    
    // 收集所有块的元数据到 Rank 0
    std::vector<BlockMeta> gather_all_block_metas();
    
    // 写入 PLT 文件头
    void write_plt_header(std::ofstream& out, int n_blocks, const std::vector<std::string>& var_names);
    
    // 写入 PLT Zone 头
    void write_plt_zone_header(std::ofstream& out, int zone_id, int ni, int nj, int nk);
    
    // 写入数据块 (double 数组)
    void write_plt_data_block(std::ofstream& out, const std::vector<double>& data);
    
private:
    std::vector<Block*>& blocks_; // 引用外部传入的 blocks
    const Config* config_;
    const MpiContext& mpi_;
    BoundaryManager boundary_manager_;

    int current_step_;
    int max_steps_;
};

} // namespace Hyres