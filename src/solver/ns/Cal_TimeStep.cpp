#include "NSKernel.h"

namespace Hyres {

void NsKernel::spectrum_tgh(Block* b) {
    // 1. 计算对流谱半径 (Inviscid Spectral Radius)
    // 对应 Fortran: call spectinv
    spectinv(b); 

    // 2. 处理粘性谱半径 (Viscous Spectral Radius)
    int nvis = config_->physics.viscous_mode;
    real_t csrv = config_->scheme.visc_spectral_radius;

    if (nvis == 1) {
        if (csrv > 0.0) {
            spectvisl(b);
        }
    } else {
        set_spectvis_to_0(b);
    }
}

void NsKernel::spectinv(Block* b) {
    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 辅助 Lambda：计算单点的谱半径
    // 参数：
    //   i, j, k: 当前网格坐标 (用于读取 U, V, W, C 和 写入 sra, srb, src)
    //   mi, mj, mk: 度量坐标 (Metric Index, 用于读取 kcx, etx 等)
    //               在物理区 mi=i, mj=j, mk=k；在边界处会被钳位(Clamp)到物理边界
    auto calc_point_spect = [&](int i, int j, int k, int mi, int mj, int mk) {
        real_t u = b->u(i, j, k);
        real_t v = b->v(i, j, k);
        real_t w = b->w(i, j, k);
        real_t c = b->c(i, j, k);

        // 1. 获取度量 (对应 getr0kec_mml 的三次调用)
        // Xi metrics (from mi, mj, mk)
        real_t kx = b->kcx(mi, mj, mk);
        real_t ky = b->kcy(mi, mj, mk);
        real_t kz = b->kcz(mi, mj, mk);
        real_t kt = b->kct(mi, mj, mk);

        // Eta metrics
        real_t ex = b->etx(mi, mj, mk);
        real_t ey = b->ety(mi, mj, mk);
        real_t ez = b->etz(mi, mj, mk);
        real_t et = b->ett(mi, mj, mk);

        // Zeta metrics
        real_t cx = b->ctx(mi, mj, mk);
        real_t cy = b->cty(mi, mj, mk);
        real_t cz = b->ctz(mi, mj, mk);
        real_t ct = b->ctt(mi, mj, mk);

        // 2. 计算逆变速度分量 (Contravariant Velocities)
        real_t cita = kx*u + ky*v + kz*w + kt;
        real_t citb = ex*u + ey*v + ez*w + et;
        real_t citc = cx*u + cy*v + cz*w + ct;

        // 3. 计算梯度模长 (Gradient Magnitudes)
        real_t cgma = std::sqrt(kx*kx + ky*ky + kz*kz);
        real_t cgmb = std::sqrt(ex*ex + ey*ey + ez*ez);
        real_t cgmc = std::sqrt(cx*cx + cy*cy + cz*cz);

        // 4. 赋值谱半径
        b->sra(i, j, k) = std::abs(cita) + c * cgma;
        b->srb(i, j, k) = std::abs(citb) + c * cgmb;
        b->src(i, j, k) = std::abs(citc) + c * cgmc;
    };

    // ========================================================
    // 1. 物理核心区域 (Inner Loop)
    // ========================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                // 物理区：度量坐标 = 网格坐标
                calc_point_spect(I, J, K, I, J, K);
            }
        }
    }

    // ========================================================
    // 2. I 方向边界 (I-Boundaries)
    // ii = max(1, min(ni, i)) -> 钳位逻辑
    // ========================================================
    int i_bounds[] = {ng - 1, ni + ng}; // 对应 Fortran 0 和 ni+1
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int I : i_bounds) {
                int J = j + ng;
                int K = k + ng;
                
                // 钳位逻辑 (Clamping)
                // 如果 I 是左 Ghost (ng-1)，取物理左边界 (ng)
                // 如果 I 是右 Ghost (ni+ng)，取物理右边界 (ni+ng-1)
                int mi = I;
                if (mi < ng) mi = ng;
                if (mi >= ni + ng) mi = ni + ng - 1;

                calc_point_spect(I, J, K, mi, J, K);
            }
        }
    }

    // ========================================================
    // 3. J 方向边界 (J-Boundaries)
    // ========================================================
    int j_bounds[] = {ng - 1, nj + ng}; 
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            for (int J : j_bounds) {
                int I = i + ng;
                int K = k + ng;

                int mj = J;
                if (mj < ng) mj = ng;
                if (mj >= nj + ng) mj = nj + ng - 1;

                calc_point_spect(I, J, K, I, mj, K);
            }
        }
    }

    // ========================================================
    // 4. K 方向边界 (K-Boundaries)
    // ========================================================
    int k_bounds[] = {ng - 1, nk + ng};
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            for (int K : k_bounds) {
                int I = i + ng;
                int J = j + ng;

                int mk = K;
                if (mk < ng) mk = ng;
                if (mk >= nk + ng) mk = nk + ng - 1;

                calc_point_spect(I, J, K, I, J, mk);
            }
        }
    }
}

void NsKernel::spectvisl(Block* b) {
    const auto& global = GlobalData::getInstance();
    real_t csrv = config_->scheme.visc_spectral_radius;
    real_t reynolds = global.reynolds;
    real_t small = 1.0e-38; 

    int ng = b->ng;
    int ni = b->ni;
    int nj = b->nj;
    int nk = b->nk;

    // 辅助 Lambda：计算单点的粘性谱半径
    // 参数：
    //   i, j, k: 目标写入坐标 (srva, srvb, srvc)
    //   mi, mj, mk: 源数据读取坐标 (vol, visl, Metrics) - 已钳位到物理区
    auto calc_visc_spect = [&](int i, int j, int k, int mi, int mj, int mk) {
        
        // 1. 获取几何与物理量 (全部来自 Clamped Index mi,mj,mk)
        // 对应 Fortran: rm_vol = r(i,j,k) * vol(ii,j,k) ? 
        // 等等！Fortran 代码里 inner loop 是 r(i,j,k)*vol(i,j,k)
        // 但在边界循环里：
        //    rm_vol = r(i,j,k) * vol(ii,j,k)  <-- 注意！r 用的是 i (Ghost), vol 用的是 ii (Clamped)
        //    vis = visl(ii,j,k)               <-- vis 用的是 ii (Clamped)
        
        // 我们必须严格区分：
        //   - r (密度) : 使用原始坐标 i,j,k
        //   - vol, visl, Metrics : 使用钳位坐标 mi,mj,mk
        
        real_t rho = b->r(i, j, k);       // Raw index
        real_t vol = b->vol(mi, mj, mk);  // Clamped index
        real_t vis = b->visl(mi, mj, mk); // Clamped index

        real_t rm_vol = rho * vol; 

        // 2. 获取度量 (Clamped Metrics)
        real_t kx = b->kcx(mi, mj, mk);
        real_t ky = b->kcy(mi, mj, mk);
        real_t kz = b->kcz(mi, mj, mk);
        
        real_t ex = b->etx(mi, mj, mk);
        real_t ey = b->ety(mi, mj, mk);
        real_t ez = b->etz(mi, mj, mk);
        
        real_t cx = b->ctx(mi, mj, mk);
        real_t cy = b->cty(mi, mj, mk);
        real_t cz = b->ctz(mi, mj, mk);

        // 3. 计算度量平方和 (ri, rj, rk)
        real_t ri = kx*kx + ky*ky + kz*kz;
        real_t rj = ex*ex + ey*ey + ez*ez;
        real_t rk = cx*cx + cy*cy + cz*cz;

        // 4. 计算系数
        // coef = 2.0 * vis / (reynolds * rm_vol + small)
        real_t denom = reynolds * rm_vol + small;
        real_t coef = 2.0 * vis / denom;
        real_t coef1 = csrv * coef;

        // 5. 赋值粘性谱半径
        b->srva(i, j, k) = coef1 * ri;
        b->srvb(i, j, k) = coef1 * rj;
        b->srvc(i, j, k) = coef1 * rk;
    };

    // ========================================================
    // 1. 物理核心区域 (Inner Loop)
    // 对应 Fortran: do k=1,nk; do j=1,nj; do i=1,ni
    // ========================================================
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                int I = i + ng;
                int J = j + ng;
                int K = k + ng;
                // 物理区：读取坐标 = 写入坐标
                calc_visc_spect(I, J, K, I, J, K);
            }
        }
    }

    // ========================================================
    // 2. I 方向边界 (I-Boundaries)
    // 对应 Fortran: do i=0, ni+1, ni+1
    // ii = max(1, min(ni, i))
    // ========================================================
    int i_bounds[] = {ng - 1, ni + ng}; 
    for (int k = 0; k < nk; ++k) {
        for (int j = 0; j < nj; ++j) {
            for (int I : i_bounds) {
                int J = j + ng;
                int K = k + ng;
                
                // Clamping
                int mi = I;
                if (mi < ng) mi = ng;
                if (mi >= ni + ng) mi = ni + ng - 1;

                calc_visc_spect(I, J, K, mi, J, K);
            }
        }
    }

    // ========================================================
    // 3. J 方向边界 (J-Boundaries)
    // 对应 Fortran: do j=0, nj+1, nj+1
    // ========================================================
    int j_bounds[] = {ng - 1, nj + ng};
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < ni; ++i) {
            for (int J : j_bounds) {
                int I = i + ng;
                int K = k + ng;

                int mj = J;
                if (mj < ng) mj = ng;
                if (mj >= nj + ng) mj = nj + ng - 1;

                calc_visc_spect(I, J, K, I, mj, K);
            }
        }
    }

    // ========================================================
    // 4. K 方向边界 (K-Boundaries)
    // 对应 Fortran: do k=0, nk+1, nk+1
    // ========================================================
    int k_bounds[] = {ng - 1, nk + ng};
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            for (int K : k_bounds) {
                int I = i + ng;
                int J = j + ng;

                int mk = K;
                if (mk < ng) mk = ng;
                if (mk >= nk + ng) mk = nk + ng - 1;

                calc_visc_spect(I, J, K, I, J, mk);
            }
        }
    }
}

void NsKernel::set_spectvis_to_0(Block* b) {
    int size_i = b->ni + 2 * b->ng;
    int size_j = b->nj + 2 * b->ng;
    int size_k = b->nk + 2 * b->ng;

    for (int k = 0; k < size_k; ++k) {
        for (int j = 0; j < size_j; ++j) {
            for (int i = 0; i < size_i; ++i) {
                b->srva(i, j, k) = 0.0;
                b->srvb(i, j, k) = 0.0;
                b->srvc(i, j, k) = 0.0;
            }
        }
    }
}

void NsKernel::localdt0(Block* b) {
    auto& global = GlobalData::getInstance();
    
    // 配置参数
    real_t cfl = config_->timeStep.cfl;
    real_t timedt_rate = config_->timeStep.dt_rate;
    real_t small = 1.0e-38;

    global.timedt = small;

    int ng = b->ng;
    
    // 对应 C++ 物理区: ng 到 ni+ng
    int is = ng; 
    int ie = b->ni + ng;
    
    int js = ng; 
    int je = b->nj + ng;
    
    int ks = ng; 
    int ke = b->nk + ng;

    for (int k = ks; k < ke; ++k) {
        for (int j = js; j < je; ++j) {
            for (int i = is; i < ie; ++i) {
                
                real_t ra = b->sra(i, j, k);
                real_t rb = b->srb(i, j, k);
                real_t rc = b->src(i, j, k);
                
                // 粘性谱半径求和
                real_t rv = b->srva(i, j, k) + b->srvb(i, j, k) + b->srvc(i, j, k);

                // 总谱半径 rabc
                real_t rabc = ra + rb + rc + rv;

                // 2. 计算时间步长
                real_t dt_val;
                
                if (timedt_rate > small) {
                    real_t vol = b->vol(i, j, k);
                    // Fortran: min(cfl/rabc, timedt_rate/vol)
                    real_t dt1 = cfl / rabc;
                    real_t dt2 = timedt_rate / vol;
                    dt_val = (dt1 < dt2) ? dt1 : dt2;
                } else {
                    dt_val = cfl / rabc;
                }

                // 3. 存入 dtdt 数组
                b->dtdt(i, j, k) = dt_val;

                // 4. 更新 timedt (统计 max(dt * vol))
                real_t val = dt_val * b->vol(i, j, k);
                if (val > global.timedt) {
                    global.timedt = val;
                }
            }
        }
    }
}

}