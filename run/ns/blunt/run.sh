#!/bin/bash
#SBATCH --job-name=SOLVER
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00             
#SBATCH --output=out%j.log
#SBATCH --partition=compute
#SBATCH --nodelist=master,node1  

# 加载环境模块
source /etc/profile.d/modules.sh
module purge
module load openmpi/5.0.5

# 设置测试可执行文件路径
OPENHITS_RUN="root@master:/home/zhanghong/HPC/MAIN-SOLVER/HYRES/bin/"
export PATH="$OPENHITS_RUN:$PATH"

# 运行 MPI+OMP 测试
mpirun --allow-run-as-root \
       -mca pml ucx \
       -mca btl ^uct \
       -x UCX_TLS=rc,dc,self \
       hyres-solver