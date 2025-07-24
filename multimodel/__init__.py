"""
MultiModel - 多模态机器学习项目

该项目提供了一个完整的多模态数据处理和融合框架，支持：
- 文本处理和理解
- 图像识别和处理
- 音频分析和处理
- 多模态特征融合
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.text_processor import TextProcessor
from .core.image_processor import ImageProcessor
from .core.audio_processor import AudioProcessor
from .core.multimodal_fusion import MultimodalFusion

__all__ = [
    "TextProcessor",
    "ImageProcessor", 
    "AudioProcessor",
    "MultimodalFusion"
]
