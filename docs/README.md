# HYRES (Hypersonic Reacting Solver)

<p align="center">
  <img src="logo.png" alt="HYRES Logo" width="600">
</p>
<p align="center">
  <b>High-Resolution | High-Performance | Hypersonic</b>
</p>
<div align="center">

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green)
![Language](https://img.shields.io/badge/C%2B%2B-17-blue)
![License](https://img.shields.io/badge/license-MIT-orange)

</div>

---

## 📖 Overview

**HYRES** is a next-generation CFD solver designed for **Direct Numerical Simulation (DNS)** and **Large Eddy Simulation (LES)** of hypersonic flows.

Unlike legacy codes, HYRES is built from the ground up for **heterogeneous computing architectures**. By leveraging modern C++ template metaprogramming and NVIDIA CUDA, it achieves extreme arithmetic intensity on GPU clusters (e.g., V100/A100) while maintaining the rigorous accuracy required for shock-capturing and thermochemical non-equilibrium flows.

**Current Status:** *Under active development (Migrating from Fortran/Legacy-WCNS to C++/CUDA).*

## 🚀 Key Features

* **GPU-Native Architecture:**
    * Optimized for massive parallelism on NVIDIA V100/A100 clusters.
    * Utilizes **MPI + CUDA** hybrid parallelism.
    * Targeting >80% peak FP64 performance via coalesced memory access patterns on structured grids.
* **High-Order Numerics:**
    * **WCNS (Weighted Compact Nonlinear Schemes)** for low-dissipation shock capturing.
    * High-order finite difference framework on multi-block structured meshes.
* **Thermochemical Non-Equilibrium:**
    * Two-Temperature Model ($T_{tr}, T_{ve}$) implementation.
    * Stiff chemistry handling via point-implicit GPU kernels.
* **Modular Design:**
    * **Zero-overhead abstraction** using C++17 `if constexpr` and templates.
    * Single codebase supporting both Calorically Perfect Gas (NS) and Non-Equilibrium models without runtime penalties.

## 🛠️ System Architecture

HYRES follows a classic decoupled pipeline optimized for HPC:

1.  **`hyres-pre` (Preprocessing):**
    * Domain decomposition (Load balancing for multi-GPU).
    * Grid metric calculation & coordinate transformation.
    * Data padding/alignment for GPU memory coalescing.
2.  **`hyres-solver` (Core):**
    * The main MPI+CUDA execution engine.
    * Runge-Kutta time integration with operator splitting.
3.  **`hyres-post` (Postprocessing):**
    * Parallel I/O using **CGNS (HDF5)**.
    * In-situ statistical analysis (Reynolds stress, Skin friction budgets).

## 💻 Building HYRES

### Prerequisites
* **Compiler:** C++17 compliant compiler (GCC 9+, Clang 10+, or Intel ICX)
* **GPU:** NVIDIA CUDA Toolkit 11.0+
* **MPI:** OpenMPI or MPICH (CUDA-aware MPI recommended)
* **Build System:** CMake 3.18+
* **I/O:** HDF5 & CGNS libraries

### Compilation
```bash
# Clone the repository
git clone [https://github.com/your-username/hyres.git](https://github.com/your-username/hyres.git)
cd hyres

# Create build directory
mkdir build && cd build

# Configure with CMake (Example for V100 Cluster)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DHYRES_ENABLE_CUDA=ON \
    -DHYRES_GPU_ARCH=sm_70 \
    -DHYRES_PRECISION=DOUBLE

# Build
make -j 8