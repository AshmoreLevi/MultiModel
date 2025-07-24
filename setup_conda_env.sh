#!/bin/bash

# MultiModel项目环境安装脚本
# 使用conda创建和配置Python环境

set -e  # 遇到错误立即退出

echo "🚀 开始设置MultiModel项目环境..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ 检测到conda环境"

# 创建conda环境
echo "📦 创建conda环境 'multimodel'..."
if conda env list | grep -q "multimodel"; then
    echo "⚠️  环境 'multimodel' 已存在，是否要重新创建？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n multimodel -y
    else
        echo "✋ 使用现有环境"
    fi
fi

if ! conda env list | grep -q "multimodel"; then
    echo "🔧 从environment.yml创建环境..."
    conda env create -f environment.yml
else
    echo "🔄 更新现有环境..."
    conda env update -f environment.yml
fi

echo "✅ Conda环境创建完成"

# 激活环境并安装项目
echo "🔌 激活环境并安装项目..."
eval "$(conda shell.bash hook)"
conda activate multimodel

# 安装项目为可编辑模式
echo "📥 安装项目包..."
pip install -e .

# 下载必要的模型和数据
echo "📚 下载必要的语言模型..."

# 下载spacy中文模型
python -c "
import spacy
import subprocess
import sys

try:
    nlp = spacy.load('zh_core_web_sm')
    print('✅ spacy中文模型已存在')
except OSError:
    print('📥 下载spacy中文模型...')
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'zh_core_web_sm'])
    print('✅ spacy中文模型下载完成')
"

# 下载nltk数据
python -c "
import nltk
import os

# 设置nltk数据路径
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

print('📥 下载NLTK数据...')
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('✅ NLTK数据下载完成')
except Exception as e:
    print(f'⚠️  NLTK数据下载可能失败: {e}')
"

# 测试安装
echo "🧪 测试安装..."
python -c "
try:
    from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion
    print('✅ 所有核心模块导入成功')
    
    # 简单测试
    text_processor = TextProcessor()
    print('✅ 文本处理器初始化成功')
    
    image_processor = ImageProcessor()
    print('✅ 图像处理器初始化成功')
    
    audio_processor = AudioProcessor()
    print('✅ 音频处理器初始化成功')
    
    fusion = MultimodalFusion()
    print('✅ 多模态融合器初始化成功')
    
except Exception as e:
    print(f'❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎉 环境安装完成！"
echo ""
echo "📋 使用说明："
echo "1. 激活环境: conda activate multimodel"
echo "2. 运行演示: python examples/demo.py"
echo "3. 使用CLI工具: multimodel --help"
echo "4. 启动Jupyter: jupyter lab"
echo ""
echo "📖 更多信息请查看 README.md"
