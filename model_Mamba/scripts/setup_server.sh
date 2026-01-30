#!/bin/bash
# PoLaRIS-Gaussian-Mamba 服务器环境配置脚本
# 用途：在 Linux + CUDA 服务器上一键配置运行环境
# 作者：PoLaRIS Team
# 日期：2026-01-30

set -e  # 遇到错误立即退出

echo "========================================"
echo "PoLaRIS-Gaussian-Mamba 服务器环境配置"
echo "========================================"

# 检查 CUDA 环境
echo ""
echo "📌 [1/5] 检查 CUDA 环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✅ CUDA 环境正常"
else
    echo "⚠️  警告: nvidia-smi 未找到，请确认 CUDA 驱动已安装"
fi

# 检查 Python 版本
echo ""
echo "📌 [2/5] 检查 Python 环境..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $PYTHON_VERSION"

REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "✅ Python 版本符合要求 (>= 3.8)"
else
    echo "❌ Python 版本过低，请使用 Python 3.8 或更高版本"
    exit 1
fi

# 检查 PyTorch + CUDA
echo ""
echo "📌 [3/5] 检查 PyTorch 安装..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')")

    echo "PyTorch 版本: $TORCH_VERSION"
    echo "CUDA 可用: $CUDA_AVAILABLE"
    echo "CUDA 版本: $CUDA_VERSION"

    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo "✅ PyTorch + CUDA 环境正常"
    else
        echo "⚠️  警告: PyTorch 未检测到 CUDA，训练将在 CPU 上进行（极慢）"
        echo "建议重新安装支持 CUDA 的 PyTorch："
        echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    fi
else
    echo "❌ PyTorch 未安装"
    echo "请先安装 PyTorch："
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# 安装必需依赖
echo ""
echo "📌 [4/5] 安装必需依赖..."
pip install opencv-python pillow tqdm einops scipy -q
echo "✅ 基础依赖安装完成"

# 安装 mamba_ssm（关键加速库）
echo ""
echo "📌 [5/5] 安装 mamba_ssm 加速库..."
if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
    echo "✅ mamba_ssm 已安装"
else
    echo "正在安装 mamba_ssm（可能需要几分钟）..."
    pip install causal-conv1d>=1.1.0 -q
    pip install mamba-ssm -q

    # 验证安装
    if python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn" 2>/dev/null; then
        echo "✅ mamba_ssm 安装成功"
    else
        echo "⚠️  mamba_ssm 安装失败，将使用 PyTorch 原生实现（速度慢 ~3x）"
        echo "可能的原因："
        echo "  1. CUDA 版本不兼容（需要 CUDA 11.7+）"
        echo "  2. gcc/g++ 编译器缺失"
        echo "  3. 网络问题"
        echo ""
        echo "手动安装命令："
        echo "  pip install causal-conv1d>=1.1.0"
        echo "  pip install mamba-ssm"
    fi
fi

# 运行单元测试
echo ""
echo "========================================"
echo "🧪 运行单元测试验证安装..."
echo "========================================"

# 保存当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# 测试各模块
echo ""
echo "测试 1/4: SS2D 组件..."
python -m model_Mamba.core.ss2d_components > /dev/null 2>&1 && echo "✅ SS2D 测试通过" || echo "❌ SS2D 测试失败"

echo "测试 2/4: PoLaRIS_Mamba 模型..."
python -m model_Mamba.core.polaris_mamba > /dev/null 2>&1 && echo "✅ 模型测试通过" || echo "❌ 模型测试失败"

echo "测试 3/4: Gaussian Focal Loss..."
python -m model_Mamba.core.loss > /dev/null 2>&1 && echo "✅ 损失函数测试通过" || echo "❌ 损失函数测试失败"

echo "测试 4/4: Gaussian 目标生成..."
python -m model_Mamba.dataset.gaussian_utils > /dev/null 2>&1 && echo "✅ 数据工具测试通过" || echo "❌ 数据工具测试失败"

# 完成
echo ""
echo "========================================"
echo "✅ 环境配置完成！"
echo "========================================"
echo ""
echo "📚 下一步："
echo "  1. 准备数据集（确保有 images/ 和 labels/ 文件夹）"
echo "  2. 运行训练："
echo "     bash scripts/train_mamba_server.sh"
echo ""
echo "📖 详细文档："
echo "  - 快速启动: MAMBA_QUICKSTART.md"
echo "  - 完整文档: model_Mamba/README.md"
echo ""
