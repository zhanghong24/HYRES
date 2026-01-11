#pragma once
#include <vector>
#include <iostream>
#include "common/Types.h"
#include "data/HyresArray.h"

namespace Hyres {

struct BoundaryPatch {
    int id;
    int type; // bctype
    int block_id;

    // 【核心修改】直接存储文件中的原始值 (1-based, 带符号)
    // 例如: start=1, end=16  或者 start=-1, end=-16
    std::array<int, 3> raw_is; // s_st
    std::array<int, 3> raw_ie; // s_ed

    // 目标块信息
    int target_block; // 建议这里还是存 0-based 以便直接索引 blocks[]，或者也存 raw_nbt
    int window_id;
    
    // 目标块的原始范围
    std::array<int, 3> raw_target_is; // t_st
    std::array<int, 3> raw_target_ie; // t_ed

    // ========================================================
    // analyze_bc 分析结果
    // ========================================================
    
    // 1. 本地几何特征
    // s_nd: 法向维度 (0=I, 1=J, 2=K)
    // s_lr: 侧边位置 (-1=Left/Bottom/Back, 1=Right/Top/Front)
    int s_nd;
    int s_lr;
    
    // 2. 对接拓扑特征 (仅当 type < 0)
    // s_t_dirction: 3x3 旋转矩阵 (Source -> Target)
    // 存储方式: matrix[source_dim][target_dim]
    int dir_matrix[3][3]; 
    
    // st_sign: 综合步长符号
    std::array<int, 3> st_sign;

    // 3. 预计算映射表 (Pre-computed Mapping Arrays)
    // 对应 Fortran: image, jmage, kmage
    // 存储的是目标块的 0-based 绝对索引
    std::vector<int> image;
    std::vector<int> jmage;
    std::vector<int> kmage;

    // 辅助：获取映射表的大小 (用于分配 vector)
    std::array<int, 3> get_size() const {
        return {
            std::abs(raw_is[0] - raw_ie[0]) + 1,
            std::abs(raw_is[1] - raw_ie[1]) + 1,
            std::abs(raw_is[2] - raw_ie[2]) + 1
        };
    }

    // ========================================================
    // MPI 通信缓冲区
    // ========================================================
    
    // --- 发送方缓冲区 (Sender / Source) ---
    HyresArray<real_t> dqv;      // 对应 Fortran: dqv (5 vars)
    HyresArray<real_t> qpv_t;    // 对应 Fortran: qpv_t (6 vars)
    HyresArray<real_t> dqv_t;    // 对应 Fortran: dqv_t (5 vars)
    
    HyresArray<real_t> vol;      // 对应 Fortran: vol (Ghost Expanded)
    HyresArray<real_t> qpv;      // 对应 Fortran: qpv (Primitive, Tangential Expanded)

    // --- 接收方缓冲区 (Receiver / Destination) ---
    HyresArray<real_t> dqvpack;
    HyresArray<real_t> qpvpack_t;
    HyresArray<real_t> dqvpack_t;
    
    HyresArray<real_t> qpvpack;

    // 辅助函数：获取 C++ (0-based) 的绝对值起始索引
    // 方便后续使用，不用到处写 abs(x)-1
    int get_abs_start(int dim) const {
        return std::abs(raw_is[dim]) - 1;
    }
    
    int get_abs_end(int dim) const {
        return std::abs(raw_ie[dim]) - 1;
    }

    // 辅助函数：判断是否逆序
    bool is_reversed(int dim) const {
        return raw_is[dim] < 0;
    }
};

class Block {
public:
    // ==========================================
    // 1. 元数据 (Metadata)
    // ==========================================
    int id;             // 全局 ID
    int ni, nj, nk;     // 物理网格尺寸 (从 Plot3D 文件读入)
    int ng;             // 幽灵层数 (Ghost Layers)

    // 基础网格坐标
    HyresArray<real_t> x, y, z;
    
    // 1. 几何度量 (Metrics)
    HyresArray<real_t> vol, volt;
    HyresArray<real_t> kcx, kcy, kcz, kct; // xi derivatives
    HyresArray<real_t> etx, ety, etz, ett; // eta derivatives
    HyresArray<real_t> ctx, cty, ctz, ctt; // zeta derivatives
    
    // 2. 原始变量 (Primitive Variables)
    HyresArray<real_t> r, u, v, w, p, t; // Rho, U, V, W, P, T
    HyresArray<real_t> c;                // Speed of Sound
    
    // 3. 守恒变量 Q (Conservative Variables)
    // 维度: (5, ni, nj, nk)
    HyresArray<real_t> q, dq;
    HyresArray<real_t> qnc, qmc; // Dual time stepping
    
    // 4. 粘性与辅助变量
    HyresArray<real_t> visl; // Laminar Viscosity
    HyresArray<real_t> vist; // Turbulent Viscosity
    
    HyresArray<real_t> sra, srb, src;
    HyresArray<real_t> srva, srvb, srvc;
    HyresArray<real_t> dtdt; 
    HyresArray<real_t> rhs1;
    
    // 化学反应与距离
    HyresArray<real_t> fs;
    HyresArray<real_t> srs;
    HyresArray<real_t> qke; 
    HyresArray<real_t> distance; // Wall distance

    // 5. 边界标志数组 (Face Flags)
    // 索引顺序: 0:I_min, 1:I_max, 2:J_min, 3:J_max, 4:K_min, 5:K_max
    std::vector<HyresArray<int>> bc_flags;
    
    // 6. 边界条件执行顺序表 (set_bc_index 用)
    std::vector<int> bc_execution_order;

    // 其他元数据
    std::string name;
    int owner_rank;
    std::vector<BoundaryPatch> boundaries;

    // ==========================================
    // 【新增】本地多点连接索引
    // ==========================================
    struct LocalSingularPoint {
        int i, j, k; // 本地坐标 (0-based, 无 Ghost 偏移，还是含 Ghost? 建议存 0-based 物理坐标)
        int buffer_seq; // 在本地发送 Buffer (dq_npp_local) 中的序列号 (0, 1, 2...)
    };
    
    // 该 Block 包含的奇异点列表
    // 用于 communicate_dq_npp 中快速从 Block 取值填入 dq_npp_local
    std::vector<LocalSingularPoint> singular_points;

public:
    // ==========================================
    // 3. 构造与生命周期
    // ==========================================
    Block(int _id, int _ni, int _nj, int _nk, int _ng) 
        : id(_id), ni(_ni), nj(_nj), nk(_nk), ng(_ng) {
        
        // 计算带 Ghost 的总维度
        int size_i = ni + 2 * ng;
        int size_j = nj + 2 * ng;
        int size_k = nk + 2 * ng;

        // --- 坐标分配 ---
        x.resize(size_i, size_j, size_k);
        y.resize(size_i, size_j, size_k);
        z.resize(size_i, size_j, size_k);
    }

    // 析构函数：HyresArray 会自动释放内存，无需手动 delete
    ~Block() = default;

    // ==========================================
    // 4. 数据填充接口 (关键)
    // ==========================================
    // 作用：将 Plot3D 读入的“纯物理网格”一维数组，填入到 HyresArray 的“中心区域”
    // raw_x/y/z 的大小必须等于 ni * nj * nk
    void fill_grid_data(const std::vector<real_t>& raw_x,
                        const std::vector<real_t>& raw_y,
                        const std::vector<real_t>& raw_z) 
    {
        // 遍历物理网格点 (i, j, k)
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    // 1. 计算 raw buffer 中的线性索引 (假设文件是 i-fastest)
                    // index = i + ni * j + ni * nj * k
                    size_t raw_idx = i + ni * j + ni * nj * k;

                    // 2. 填充到 Block 的 HyresArray 中
                    // 注意加上偏移量 ng
                    x(i + ng, j + ng, k + ng) = raw_x[raw_idx];
                    y(i + ng, j + ng, k + ng) = raw_y[raw_idx];
                    z(i + ng, j + ng, k + ng) = raw_z[raw_idx];
                }
            }
        }
    }

    // ========================================================
    // 只释放数据负载，保留拓扑结构
    // ========================================================
    void free_payload() {
        // 释放网格坐标
        x.clear(); y.clear(); z.clear();
        
        // 释放流场数据
        r.clear(); u.clear(); v.clear(); w.clear(); p.clear(); t.clear();
        q.clear();

        // 注意：绝对不要 clear boundaries！
        // exchange_bc 需要用到 boundaries 中的几何信息来生成 Tag
    }
};

} // namespace Hyres