#!/bin/bash

# 遇到任何错误立即停止脚本 (防止错误滚雪球)
set -e

# 1. 加载环境 (根据你的服务器配置)
module load openmpi/5.0.5 cmake/3.29.8

# 2. 关键修改：使用 -p 参数
# 如果 build 不存在则创建；如果存在则什么都不做，且不报错
mkdir -p build

# 3. 进入目录
cd build

# 4. CMake 配置
# 第一次运行会生成 Makefile，后续运行只会更新配置，非常快
echo ">>> Configuring..."
cmake .. -DHYRES_ARCH=CPU

# 5. 增量编译
echo ">>> Compiling..."
make -j 64

echo ">>> Build Success!"