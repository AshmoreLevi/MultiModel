"""
基础测试模块

测试各个核心组件的基本功能
"""

import pytest
import numpy as np
from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion


class TestTextProcessor:
    """测试文本处理器"""
    
    def test_text_cleaning(self) -> None:
        """测试文本清理功能"""
        processor = TextProcessor()
        text = "这是一个测试文本！！！   包含特殊字符@#$%"
        cleaned = processor.clean_text(text)
        
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
    
    def test_chinese_tokenization(self) -> None:
        """测试中文分词"""
        processor = TextProcessor()
        text = "我喜欢机器学习和人工智能"
        tokens = processor.tokenize_chinese(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_text_statistics(self) -> None:
        """测试文本统计功能"""
        processor = TextProcessor()
        text = "这是一个简单的测试文本"
        stats = processor.get_text_statistics(text)
        
        assert "original_length" in stats
        assert "word_count" in stats
        assert "unique_words" in stats
        assert stats["word_count"] > 0


class TestImageProcessor:
    """测试图像处理器"""
    
    def test_image_preprocessing(self) -> None:
        """测试图像预处理"""
        processor = ImageProcessor()
        # 创建模拟图像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        processed = processor.preprocess_image(image)
        
        assert "original_shape" in processed
        assert "resized" in processed
        assert "normalized" in processed
    
    def test_color_analysis(self) -> None:
        """测试颜色分析"""
        processor = ImageProcessor()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        colors = processor.analyze_color(image)
        
        assert "mean_rgb" in colors
        assert "dominant_colors" in colors
    
    def test_object_detection(self) -> None:
        """测试对象检测"""
        processor = ImageProcessor()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        objects = processor.detect_objects(image)
        
        assert isinstance(objects, list)


class TestAudioProcessor:
    """测试音频处理器"""
    
    def test_audio_preprocessing(self) -> None:
        """测试音频预处理"""
        processor = AudioProcessor()
        # 创建模拟音频数据（3秒的正弦波）
        duration = 3
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
        
        processed = processor.preprocess_audio(audio_data)
        
        assert "original_length" in processed
        assert "duration" in processed
        assert "normalized" in processed
    
    def test_mfcc_extraction(self) -> None:
        """测试MFCC特征提取"""
        processor = AudioProcessor()
        audio_data = np.random.randn(22050)  # 1秒随机音频
        
        mfcc = processor.extract_mfcc(audio_data)
        
        assert isinstance(mfcc, np.ndarray)
        if mfcc.size > 0:  # 如果成功提取特征
            assert mfcc.ndim == 2
    
    def test_audio_quality_analysis(self) -> None:
        """测试音频质量分析"""
        processor = AudioProcessor()
        audio_data = np.random.randn(22050)  # 1秒随机音频
        
        quality = processor.analyze_audio_quality(audio_data)
        
        assert "energy" in quality
        assert "dynamic_range" in quality


class TestMultimodalFusion:
    """测试多模态融合"""
    
    def test_concatenation_fusion(self) -> None:
        """测试连接融合"""
        fusion = MultimodalFusion(fusion_method="concatenation")
        
        text_features = np.random.randn(128)
        image_features = np.random.randn(256)
        audio_features = np.random.randn(64)
        
        result = fusion.fuse_modalities(
            text_features=text_features,
            image_features=image_features,
            audio_features=audio_features
        )
        
        assert result.get("success") is not False
        if "fused_features" in result:
            assert result["fused_features"].size > 0
    
    def test_attention_fusion(self) -> None:
        """测试注意力融合"""
        fusion = MultimodalFusion(fusion_method="attention")
        
        text_features = np.random.randn(100)
        image_features = np.random.randn(200)
        
        result = fusion.fuse_modalities(
            text_features=text_features,
            image_features=image_features
        )
        
        assert result.get("success") is not False
    
    def test_similarity_computation(self) -> None:
        """测试相似度计算"""
        fusion = MultimodalFusion()
        
        features1 = np.random.randn(100)
        features2 = np.random.randn(100)
        
        similarity = fusion.compute_similarity(features1, features2, method="cosine")
        
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1
    
    def test_modality_contribution_analysis(self) -> None:
        """测试模态贡献度分析"""
        fusion = MultimodalFusion()
        
        features_list = [
            np.random.randn(50),
            np.random.randn(100),
            np.random.randn(75)
        ]
        modality_names = ["text", "image", "audio"]
        
        contribution = fusion.analyze_modality_contribution(features_list, modality_names)
        
        assert len(contribution) <= len(modality_names)
        for name in modality_names:
            if name in contribution:
                assert "variance" in contribution[name]


def test_device_detection() -> None:
    """测试设备检测（特别是Apple Silicon MPS）"""
    import torch
    
    # 测试设备检测逻辑
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon MPS 可用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ CUDA 可用")
    else:
        device = torch.device("cpu")
        print("✅ 使用 CPU")
    
    # 验证tensor可以在设备上运行
    test_tensor = torch.randn(10, 10).to(device)
    assert test_tensor.device.type == device.type


if __name__ == "__main__":
    # 运行基本测试
    pytest.main([__file__, "-v"])
