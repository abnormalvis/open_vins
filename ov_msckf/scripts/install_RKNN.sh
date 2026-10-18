#!/bin/bash
# RKNN-Lite 单独安装脚本 for Orange Pi CM5 (RK3588S)
# 使用方法: chmod +x install_rknn_lite.sh && ./install_rknn_lite.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查是否在 ARM64 系统
if [[ $(uname -m) != "aarch64" ]]; then
    log_error "此脚本仅适用于 ARM64 (aarch64) 架构的开发板。"
    exit 1
fi

# 检查 Python 版本是否为 3.8
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.8" ]]; then
    log_warn "当前 Python 版本为 $PYTHON_VERSION，推荐使用 Python 3.8。"
    read -p "是否继续安装？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 pip3 是否可用
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 未安装，请先安装 python3-pip。"
    exit 1
fi

# 安装依赖（仅 numpy，如果未安装）
log_info "检查并安装 numpy..."
pip3 show numpy &> /dev/null || pip3 install numpy

# 克隆 rknn-toolkit2 仓库（如果已存在则拉取最新）
cd ~
if [ -d "rknn-toolkit2" ]; then
    log_info "rknn-toolkit2 目录已存在，拉取最新代码..."
    cd rknn-toolkit2
    git pull
else
    log_info "克隆 rknn-toolkit2 仓库..."
    git clone https://github.com/rockchip-linux/rknn-toolkit2.git
    cd rknn-toolkit2
fi

# 进入 packages 目录
cd rknn_toolkit_lite2/packages

# 查找对应 Python 3.8 的 wheel 文件
RKNN_WHEEL=$(ls rknn_toolkit_lite2-1.6.0-cp38-cp38-linux_aarch64.whl 2>/dev/null || echo "")
if [ -z "$RKNN_WHEEL" ]; then
    log_error "未找到 rknn_toolkit_lite2 的 wheel 文件。请检查仓库内容或版本。"
    exit 1
fi

log_info "安装 RKNN-Lite: $RKNN_WHEEL"
pip3 install --force-reinstall "$RKNN_WHEEL"

# 验证安装
log_info "验证 RKNN-Lite 安装..."
python3 -c "from rknnlite.api import RKNNLite; print('RKNN-Lite 版本:', RKNNLite().version())"

log_info "RKNN-Lite 安装完成！"
log_info "可以使用以下命令检查 NPU 状态: cat /sys/kernel/debug/rknpu/load"