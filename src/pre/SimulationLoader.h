#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "data/Block.h"
#include "common/MpiContext.h"
#include "common/Config.h" 
#include "physics/PhysicsInitializer.h"

namespace Hyres {

struct BlockGeomMeta {
    int id;
    int ni, nj, nk;
    int owner_rank; // 对应 mb_pids
};

class SimulationLoader {
public:
    // 【修改】这里接收 const Config&
    static std::vector<Block*> load(const Config* config, const MpiContext& mpi);

private:
    // 核心读取函数 (Rank 0 执行)
    static void read_plot3d_grid(std::vector<Block*>& blocks, 
                                const std::string& filename, 
                                int ng);

    // 核心读取函数 (Rank 0 执行)
    static void read_bc_parallel(std::vector<Block*>& blocks, 
                                const std::string& filename,
                                const MpiContext& mpi);

    // 【修改】只负责分发网格
    static void distribute_grid(std::vector<Block*>& blocks, const MpiContext& mpi, int ng);

    // 分发边界条件 (全场广播)
    static void distribute_bc(std::vector<Block*>& blocks, const MpiContext& mpi, int ng);

    static void analyze_bc_connect(BoundaryPatch& patch, int ndim);

    static void analyze_bc(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void get_local_info(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void alloc_bc_comm_array(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void cleanup_blocks(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void set_bc_index(std::vector<Block*>& blocks);

    static void fill_bc_flag(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void allocate_other_variable(std::vector<Block*>& blocks, const MpiContext& mpi);

    // 计算网格导数的主入口
    static void set_grid_derivative(std::vector<Block*>& blocks, const MpiContext& mpi);
    
    // 检查网格质量 (负体积检查) - 下一步实现
    static void check_grid_derivative(std::vector<Block*>& blocks, const MpiContext& mpi);

    // 单块计算驱动
    static void grid_derivative_nb(Block* b);

    // GCL 核心算法 (对应 GRID_DERIVATIVE_gcl)
    static void compute_metrics_gcl(Block* b);

    static void value_half_node(int n, int nmax, int len, 
                                const std::vector<double>& f, 
                                std::vector<double>& fh);

    static void flux_dxyz(int n, int nmax, int len, 
                                const std::vector<double>& fh, 
                                std::vector<double>& df);

    static void check_load_balance(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void initialization(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi);

    static void init_nstart0(std::vector<Block*>& blocks, const MpiContext& mpi);

    static void init_nstart1(std::vector<Block*>& blocks, const MpiContext& mpi);
    
    static void unpack_block_data(Block* b, const std::vector<double>& buffer);

    static void initialize_temperature(std::vector<Block*>& blocks, const Config* config, const MpiContext& mpi);

    static void build_nppos_list_ex(std::vector<Block*>& blocks, const MpiContext& mpi);
    
    // ===============================================
    // 二进制文件读取助手 (Helper Functions)
    // ===============================================
    
    // 跳过 Record Header/Footer (通常是 4 字节)
    static void skip_marker(std::ifstream& ifs) {
        ifs.ignore(4);
    }

    // 读取一个 int (自动处理 Header/Footer)
    static void read_int(std::ifstream& ifs, int& value) {
        skip_marker(ifs);
        ifs.read(reinterpret_cast<char*>(&value), sizeof(int));
        skip_marker(ifs);
    }

    // 读取 3 个 int (自动处理 Header/Footer)
    static void read_3ints(std::ifstream& ifs, int& v1, int& v2, int& v3) {
        skip_marker(ifs);
        ifs.read(reinterpret_cast<char*>(&v1), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&v2), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&v3), sizeof(int));
        skip_marker(ifs);
    }
};

} // namespace Hyres