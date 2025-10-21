#!/bin/bash

# CogVideoX1.5 视频遮挡工具 - 环境安装脚本
# 适用于 Python 3.10-3.12

echo "=========================================="
echo "CogVideoX1.5 环境配置安装"
echo "=========================================="
echo ""

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ 当前Python版本: $PYTHON_VERSION"
echo ""

# 创建必要的目录
echo "📁 创建项目目录..."
mkdir -p /root/autodl-tmp/sida/occlusion_new/models
mkdir -p /root/autodl-tmp/sida/urban_videos/with_occlusion
echo "✓ 目录创建完成"
echo ""

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip -q
echo "✓ pip升级完成"
echo ""

# 安装PyTorch (CUDA 12.1)
echo "🔥 安装PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch安装完成"
echo ""

# 安装核心依赖
echo "📚 安装核心依赖库..."
pip install diffusers==0.31.0
pip install transformers
pip install accelerate
echo "✓ 核心库安装完成"
echo ""

# 安装图像处理库
echo "🖼️  安装图像处理库..."
pip install opencv-python
pip install pillow
pip install numpy
pip install imageio
pip install imageio-ffmpeg
echo "✓ 图像处理库安装完成"
echo ""

# 可选：安装优化库（用于进一步提升性能）
echo "⚡ 安装性能优化库（可选）..."
pip install xformers
pip install invisible-watermark>=0.2.0
echo "✓ 优化库安装完成"
echo ""

# 验证安装
echo "=========================================="
echo "🔍 验证安装"
echo "=========================================="
python -c "
import torch
import diffusers
print(f'✓ PyTorch版本: {torch.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'✓ 显存大小: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
print(f'✓ Diffusers版本: {diffusers.__version__}')
"
echo ""

echo "=========================================="
echo "✅ 环境配置完成!"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 将Python脚本保存为: /root/autodl-tmp/sida/occlusion_new/video_occlusion_optimized.py"
echo "2. 确保视频文件在: /root/autodl-tmp/sida/urban_videos/original/"
echo "3. 运行: python video_occlusion_optimized.py"
echo ""
echo "注意："
echo "- 首次运行会自动下载模型（约15-20GB）"
echo "- 下载可能需要10-30分钟，取决于网络速度"
echo "- 模型会保存在: /root/autodl-tmp/sida/occlusion_new/models"
echo ""