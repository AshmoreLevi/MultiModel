@echo off
REM MultiModel项目环境安装脚本 (Windows版本)
REM 使用conda创建和配置Python环境

echo 🚀 开始设置MultiModel项目环境...

REM 检查conda是否安装
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 错误: 未找到conda，请先安装Anaconda或Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ 检测到conda环境

REM 创建conda环境
echo 📦 创建conda环境 'multimodel'...
conda env list | findstr "multimodel" >nul
if %ERRORLEVEL% EQU 0 (
    echo ⚠️  环境 'multimodel' 已存在，是否要重新创建？(y/N^)
    set /p response=
    if /i "%response%"=="y" (
        echo 🗑️  删除现有环境...
        conda env remove -n multimodel -y
    ) else (
        echo ✋ 使用现有环境
    )
)

conda env list | findstr "multimodel" >nul
if %ERRORLEVEL% NEQ 0 (
    echo 🔧 从environment.yml创建环境...
    conda env create -f environment.yml
) else (
    echo 🔄 更新现有环境...
    conda env update -f environment.yml
)

echo ✅ Conda环境创建完成

REM 激活环境并安装项目
echo 🔌 激活环境并安装项目...
call conda activate multimodel

REM 安装项目为可编辑模式
echo 📥 安装项目包...
pip install -e .

REM 下载必要的模型和数据
echo 📚 下载必要的语言模型...

REM 下载spacy中文模型
python -c "import spacy; import subprocess; import sys; spacy.load('zh_core_web_sm') if True else subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'zh_core_web_sm'])" 2>nul || (
    echo 📥 下载spacy中文模型...
    python -m spacy download zh_core_web_sm
    echo ✅ spacy中文模型下载完成
)

REM 下载nltk数据
python -c "import nltk; import os; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); print('✅ NLTK数据下载完成')"

REM 测试安装
echo 🧪 测试安装...
python -c "from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion; print('✅ 所有核心模块导入成功')"

echo.
echo 🎉 环境安装完成！
echo.
echo 📋 使用说明：
echo 1. 激活环境: conda activate multimodel
echo 2. 运行演示: python examples/demo.py
echo 3. 使用CLI工具: multimodel --help
echo 4. 启动Jupyter: jupyter lab
echo.
echo 📖 更多信息请查看 README.md
pause
