#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <iomanip> 
#include "data/Block.h"
#include "common/Config.h"
#include "common/MpiContext.h"
#include "data/GlobalData.h"
#include "spdlog/spdlog.h"

namespace Hyres {

class NsKernel {
public:
    // ========================================================
    // 编译期常量：变量数量 (Density, u, v, w, Energy)
    // ========================================================
    static constexpr int n_vars = 5;

    explicit NsKernel(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi);
    ~NsKernel();

    // ========================================================
    // 核心接口 (由 SolverDriver 调用)
    // ========================================================

    // 1. 边界条件处理 (包含 MPI 交换和物理边界)
    void apply_boundary();

    // 2. 计算时间步长 (dt)
    void compute_time_step();

    // 3. 计算右端项 (RHS = Inviscid + Viscous)
    void compute_rhs();

    // 4. 隐式时间推进 (LHS, LU-SGS)
    void compute_lhs();

    // 5. 残差检查与打印
    void check_residual(int step);

    // 6. 气动力监测
    void calculate_aerodynamic_forces(int step);

    // 7. 输出模块
    void output_solution(int step);

    // 8. 重启动输出模块
    void output_restart(int step);

private:

    // ========================================================
    // 气动力计算
    // ========================================================

    struct AeroCoeffs {
        real_t Fx = 0.0, Fy = 0.0, Fz = 0.0;
        real_t Mx = 0.0, My = 0.0, Mz = 0.0;
    };

    void integrate_wall_forces(Block* b, AeroCoeffs& sum, real_t p_inf);
    
    // ========================================================
    // 状态更新与准备
    // ========================================================

    void update_derived_variables();
    void update_thermodynamics(Block* b);
    void update_laminar_viscosity(Block* b);
    void update_sound_speed(Block* b);
    void update_conservative_vars(Block* b);
    void reset_residuals(Block* b);

    // ========================================================
    // 时间步长
    // ========================================================

    void spectrum_tgh(Block* b);
    void spectinv(Block* b);
    void spectvisl(Block* b);
    void set_spectvis_to_0(Block* b);
    void localdt0(Block* b);

    // ========================================================
    // MPI 通信与数据同步
    // ========================================================

    void exchange_bc(); 
    void communicate_dq_npp();
    void communicate_pv_npp();
    void exchange_bc_dq_vol();
    void exchange_bc_pv_vol();
    void boundary_match_dq_2pm();
    void boundary_match_pv_2pm();
    void boundary_match_dq_3pp();
    void boundary_match_pv_3pp();

    // ========================================================
    // 粘性通量计算 (Viscous Flux)
    // ========================================================

    void calculate_viscous_rhs();
    void compute_block_viscous_rhs(Block* b);

    // ========================================================
    // 无粘通量计算 (Inviscid Flux)
    // ========================================================

    void calculate_inviscous_rhs();
    void compute_block_inviscous_rhs(Block* b);
    
    // ========================================================
    // 隐式求解 (Implicit LHS)
    // ========================================================

    void set_boundary_dq_zero(Block* b);
    void solve_gauss_seidel_single(Block* b);
    void store_rhs_nb(Block* b, std::vector<real_t>& rhs_nb);
    void lusgs_l(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta, real_t extra);
    void lusgs_u(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta, real_t extra);
    void gs_pr_l(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta);
    void gs_pr_u(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta);
    void matrix_vector_product_std(const real_t* prim, const real_t* metrics, const real_t* dq, 
                                   real_t* f_out, real_t rad, int npn, real_t gamma);
    void update_conservatives(Block* b);
    void lusgs_rb_sweep(Block* b, const std::vector<real_t>& rhs_nb, real_t wmig, real_t beta, int color);
    // ========================================================
    // 边界条件处理函数
    // ========================================================
    
    struct BoundaryManager{

        const Config* config_; 
        std::vector<Block*>& blocks_;

        // 构造函数：初始化这些成员
        BoundaryManager(const Config* config, std::vector<Block*>& blocks) 
            : config_(config), blocks_(blocks) {}

        // 唯一接口：执行所有块的边界填充（由apply_boundary调用）
        void boundary_sequence(Block* b);

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

        void dif_average(Block* b, const BoundaryPatch& patch);
    };
    
    // ========================================================
    // 数学辅助函数 (底层算法)
    // ========================================================

    void uvwt_der_4th_virtual(Block* b, HyresArray<real_t>& duvwt);
    void uvwt_der_4th_half_virtual(Block* b, HyresArray<real_t>& duvwt_mid);
    
    void compute_duvwt_node_line_virtual(int nmax, int ni, const std::vector<std::array<real_t, 4>>& uvwt, std::vector<std::array<real_t, 4>>& duvwt, int offset);
    void compute_duvwt_half_line_virtual(int nmax, int ni, const std::vector<std::array<real_t, 4>>& uvwt, std::vector<std::array<real_t, 4>>& duvwt, int offset);
    void compute_value_line_half_virtual(int nmax, int ni, const std::vector<std::vector<real_t>>& q, std::vector<std::vector<real_t>>& q_half, int offset);
    void compute_value_half_node_ijk(int nmax, int ni, int n_comps, int ip, int jp, int kp, const std::vector<std::vector<real_t>>& q, std::vector<std::vector<real_t>>& q_half);
    void compute_value_half_node(int nmax, int ni, int n_comps, const std::vector<std::vector<real_t>>& q, std::vector<std::vector<real_t>>& q_half);
    void compute_duvwt_dxyz(int nmax, int ni, const std::vector<std::vector<real_t>>& duvwt, const std::vector<std::vector<real_t>>& kxyz, const std::vector<real_t>& vol, std::vector<std::vector<real_t>>& duvwtdxyz);
    void compute_flux_vis_line_new(int nmax, int ni, const std::vector<std::vector<real_t>>& uvwt, const std::vector<std::vector<real_t>>& duvwtdxyz, const std::vector<std::vector<real_t>>& kxyz, int idx_nx, int idx_ny, int idx_nz, const std::vector<real_t>& vslt1, const std::vector<real_t>& vslt2, std::vector<std::vector<real_t>>& fv);
    void compute_flux_dxyz(int nmax, int ni, int nl, const std::vector<std::vector<real_t>>& f, std::vector<std::vector<real_t>>& df);
    void compute_value_half_node_scalar(int nmax, int ni, const std::vector<real_t>& q, std::vector<real_t>& q_half);
    
    // 通量计算
    void flux_line_wcns_roe(int ni, const std::vector<std::array<real_t, 6>>& q_line_prim, const std::vector<std::array<real_t, 6>>& q_line_cons, const std::vector<std::array<real_t, 5>>& trxyz, real_t efix, std::vector<std::array<real_t, 5>>& fc);
    void flux_roe_kernel(const real_t* ql_prim, const real_t* qr_prim, real_t nx, real_t ny, real_t nz, real_t gamma, real_t entropy_fix, real_t* f_interface);

    // 输出模块：VTK XML Structured Grid (.vts) + MultiBlock (.vtm)
    void write_block_vts(Block* b, const std::string& filename);
    void write_global_vtm(const std::string& filename, const std::string& step_str, int total_blocks);

private:
    std::vector<Block*>& blocks_; 
    const Config* config_;
    const MpiContext& mpi_;
    
    // 物理边界管理器 (处理固壁、远场等)
    BoundaryManager boundary_manager_;
};

} // namespace Hyres