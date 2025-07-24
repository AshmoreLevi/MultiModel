#!/bin/bash

# Mac M系列芯片多模态项目安装脚本
# 自动检测并配置最佳环境

set -e  # 遇到错误立即退出

echo "🍎 Mac M系列芯片多模态项目安装脚本"
echo "========================================="

# 检查是否为Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "✅ 检测到Apple Silicon (M系列芯片)"
    ARCH="arm64"
else
    echo "⚠️  检测到Intel芯片，建议在Apple Silicon设备上运行"
    ARCH="x86_64"
fi

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 未检测到conda，请先安装Miniconda或Anaconda"
    echo "📥 下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ 检测到conda: $(conda --version)"

# 检查environment.yml是否存在
if [[ ! -f "environment.yml" ]]; then
    echo "❌ 未找到environment.yml文件"
    exit 1
fi

# 创建conda环境
echo "📦 创建conda环境..."
if conda env list | grep -q "multimodel"; then
    echo "⚠️  环境 'multimodel' 已存在，是否删除并重新创建? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n multimodel -y
    else
        echo "💡 使用现有环境，跳过创建步骤"
        conda activate multimodel
        echo "✅ 环境激活成功"
        exit 0
    fi
fi

echo "🚀 创建新的conda环境..."
conda env create -f environment.yml

# 激活环境
echo "🔄 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multimodel

# 验证PyTorch安装
echo "🔍 验证PyTorch安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'MPS可用: {torch.backends.mps.is_available()}')
print(f'MPS构建: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ 可以使用Apple Silicon GPU加速')
else:
    print('⚠️  将使用CPU运算')
"

# 安装项目
echo "📥 安装项目..."
pip install -e .

# 下载必要的模型和数据
echo "📚 下载必要的NLP模型..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK数据下载完成')
except:
    print('⚠️  NLTK数据下载失败，可稍后手动下载')
"

# 测试安装
echo "🧪 测试安装..."
python -c "
try:
    from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion
    print('✅ 所有模块导入成功')
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    exit(1)
"

echo ""
echo "🎉 安装完成！"
echo ""
echo "💡 下一步："
echo "1. 激活环境: conda activate multimodel"
echo "2. 运行演示: python examples/demo.py"
echo "3. 或使用命令行工具: multimodel demo"
echo ""
echo "📖 更多信息请查看 README.md"
