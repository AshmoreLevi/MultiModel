# MultiModel - 多模态机器学习项目

一个强大的多模态机器学习框架，支持文本、图像、音频等多种数据模态的处理、特征提取和融合。

## 🚀 项目特性

- **多模态支持**: 集成文本、图像、音频处理能力
- **特征提取**: 使用深度学习模型提取高质量特征
- **模态融合**: 支持多种融合策略（连接、注意力、加权、PCA等）
- **易于使用**: 提供简洁的API和命令行工具
- **可扩展性**: 模块化设计，便于添加新的模态和算法
- **Apple Silicon优化**: 原生支持Mac M系列芯片，使用MPS加速

## 🍎 Mac M系列芯片用户

**快速开始**: 使用我们的一键安装脚本

```bash
git clone <your-repo-url>
cd MultiModel
./install_mac_m.sh
```

📖 [完整的Mac M系列使用指南](MAC_M_SERIES_GUIDE.md)

## 📦 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- 对于Mac M系列芯片：推荐使用conda进行环境管理
- 其他依赖见 `requirements.txt` 或 `environment.yml`

### 安装步骤

#### 方法一：使用conda（推荐，特别是Mac M系列芯片）

1. 克隆项目
```bash
git clone <your-repo-url>
cd MultiModel
```

2. 创建conda环境
```bash
# 使用environment.yml创建环境（推荐）
conda env create -f environment.yml

# 激活环境
conda activate multimodel
```

3. 安装项目
```bash
pip install -e .
```

#### 方法二：使用pip（传统方式）

1. 克隆项目
```bash
git clone <your-repo-url>
cd MultiModel
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 安装项目
```bash
pip install -e .
```

#### Mac M系列芯片特别说明

- 已针对Apple Silicon进行优化
- 自动使用Metal Performance Shaders (MPS)进行GPU加速
- 移除了CUDA依赖，改用Apple Silicon原生支持

## 🎯 快速开始

### Python API 使用

```python
from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion

# 文本处理
text_processor = TextProcessor()
text_features = text_processor.extract_features("这是一段示例文本")

# 图像处理
image_processor = ImageProcessor()
image = image_processor.load_image("path/to/image.jpg")
image_features = image_processor.extract_features(image)

# 音频处理
audio_processor = AudioProcessor()
audio_data, sr = audio_processor.load_audio("path/to/audio.wav")
audio_features = audio_processor.extract_comprehensive_features(audio_data)

# 多模态融合
fusion = MultimodalFusion(fusion_method="attention")
result = fusion.fuse_modalities(
    text_features=text_features,
    image_features=image_features,
    audio_features=audio_features['mfcc'].mean(axis=1)
)

print(f"融合特征维度: {result['output_shape']}")
```

### 命令行工具使用

```bash
# 处理文本
multimodel process-text "你的文本内容" --output result.json

# 处理图像
multimodel process-image path/to/image.jpg --output ./results/

# 处理音频
multimodel process-audio path/to/audio.wav --duration 10 --output ./results/

# 多模态融合
multimodel fuse --text "文本内容" --image image.jpg --audio audio.wav --method attention --output fusion_result.json

# 运行演示
multimodel demo
```

## 📁 项目结构

```
MultiModel/
├── multimodel/              # 主要代码包
│   ├── __init__.py         # 包初始化
│   ├── core/               # 核心模块
│   │   ├── text_processor.py      # 文本处理
│   │   ├── image_processor.py     # 图像处理
│   │   ├── audio_processor.py     # 音频处理
│   │   └── multimodal_fusion.py   # 多模态融合
│   └── cli.py              # 命令行接口
├── examples/               # 示例代码
│   └── demo.py            # 演示程序
├── tests/                 # 测试文件
├── requirements.txt       # 依赖包列表
├── setup.py              # 安装配置
└── README.md             # 项目说明
```

## 🔧 核心模块

### 文本处理器 (TextProcessor)

- **功能**: 文本清理、中文分词、特征提取、统计分析
- **支持模型**: BERT、RoBERTa等Transformer模型
- **特征**: 支持中英文混合文本处理

```python
processor = TextProcessor(model_name="bert-base-chinese")
features = processor.extract_features("你的文本")
stats = processor.get_text_statistics("你的文本")
```

### 图像处理器 (ImageProcessor)

- **功能**: 图像预处理、特征提取、对象检测、颜色分析
- **支持模型**: ResNet、VGG等CNN模型
- **特征**: 自动图像增强和特征标准化

```python
processor = ImageProcessor(model_name="resnet50")
image = processor.load_image("image.jpg")
features = processor.extract_features(image)
objects = processor.detect_objects(image)
```

### 音频处理器 (AudioProcessor)

- **功能**: 音频预处理、MFCC提取、频谱分析、节奏检测
- **支持格式**: WAV、MP3、FLAC等常见音频格式
- **特征**: 梅尔频谱图、色度特征、节拍跟踪

```python
processor = AudioProcessor(sample_rate=22050)
audio_data, sr = processor.load_audio("audio.wav")
features = processor.extract_comprehensive_features(audio_data)
```

### 多模态融合器 (MultimodalFusion)

- **融合方法**:
  - `concatenation`: 简单特征连接
  - `attention`: 注意力机制融合
  - `weighted`: 加权融合
  - `pca`: PCA降维融合

```python
fusion = MultimodalFusion(fusion_method="attention")
result = fusion.fuse_modalities(
    text_features=text_feat,
    image_features=image_feat,
    audio_features=audio_feat
)
```

## 📊 示例应用

### 1. 情感分析
结合文本内容、图像情绪和音频语调进行综合情感分析

### 2. 内容理解
多模态内容检索和相似度匹配

### 3. 创意生成
基于多模态输入的创意内容生成

### 4. 质量评估
多维度内容质量自动评估

## 🧪 测试

运行测试套件：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_text_processor.py

# 运行演示程序
python examples/demo.py
```

## 📈 性能优化

- **GPU加速**: 自动检测CUDA并使用GPU加速
- **批处理**: 支持批量数据处理
- **内存优化**: 大文件流式处理
- **缓存机制**: 模型和特征缓存

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

本项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系我们

- 项目链接: [https://github.com/yourusername/multimodel](https://github.com/yourusername/multimodel)
- 问题反馈: [Issues](https://github.com/yourusername/multimodel/issues)

## 🔗 相关资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Transformers 库](https://huggingface.co/transformers/)
- [OpenCV 文档](https://docs.opencv.org/)
- [Librosa 文档](https://librosa.org/)

## 📋 更新日志

### v0.1.0 (2024-12-19)
- ✨ 初始版本发布
- 🎯 支持文本、图像、音频处理
- 🔀 实现多种模态融合方法
- 🛠️ 提供命令行工具
- 📖 完整的文档和示例

---

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**
