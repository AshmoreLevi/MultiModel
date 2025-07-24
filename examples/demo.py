"""
多模态项目示例代码

演示如何使用文本、图像、音频处理器和多模态融合功能
"""

import numpy as np
import os
from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion


def demo_text_processing() -> None:
    """演示文本处理功能"""
    print("=== 文本处理演示 ===")
    
    # 初始化文本处理器
    text_processor = TextProcessor()
    
    # 示例文本
    texts = [
        "这是一个多模态机器学习项目的示例文本。",
        "我们可以处理中文和英文文本，提取特征进行分析。",
        "Text processing includes tokenization, feature extraction, and analysis."
    ]
    
    for i, text in enumerate(texts):
        print(f"\n文本 {i+1}: {text}")
        
        # 清理文本
        cleaned = text_processor.clean_text(text)
        print(f"清理后: {cleaned}")
        
        # 中文分词
        tokens = text_processor.tokenize_chinese(text)
        print(f"分词结果: {tokens}")
        
        # 获取统计信息
        stats = text_processor.get_text_statistics(text)
        print(f"统计信息: {stats}")
    
    # 批量处理
    print("\n=== 批量文本处理 ===")
    batch_results = text_processor.batch_process(texts)
    print(f"批量处理完成，处理了 {len(batch_results['cleaned_texts'])} 个文本")


def demo_image_processing() -> None:
    """演示图像处理功能"""
    print("\n=== 图像处理演示 ===")
    
    # 初始化图像处理器
    image_processor = ImageProcessor()
    
    # 创建示例图像（随机图像）
    print("创建示例图像数据...")
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 预处理图像
    processed = image_processor.preprocess_image(sample_image)
    print(f"原始图像形状: {processed['original_shape']}")
    print(f"调整后形状: {processed['resized'].shape}")
    
    # 颜色分析
    color_analysis = image_processor.analyze_color(sample_image)
    print(f"主要颜色数量: {len(color_analysis.get('dominant_colors', []))}")
    
    # 对象检测
    objects = image_processor.detect_objects(sample_image)
    print(f"检测到 {len(objects)} 个对象")


def demo_audio_processing() -> None:
    """演示音频处理功能"""
    print("\n=== 音频处理演示 ===")
    
    # 初始化音频处理器
    audio_processor = AudioProcessor()
    
    # 创建示例音频数据（正弦波）
    print("创建示例音频数据...")
    duration = 3  # 3秒
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4音符
    sample_audio = np.sin(2 * np.pi * frequency * t)
    
    # 预处理音频
    processed = audio_processor.preprocess_audio(sample_audio)
    print(f"音频时长: {processed['duration']:.2f} 秒")
    print(f"原始长度: {processed['original_length']} 样本")
    
    # 提取音频特征
    print("提取音频特征...")
    features = audio_processor.extract_comprehensive_features(sample_audio)
    
    if 'mfcc_mean' in features:
        print(f"MFCC特征维度: {len(features['mfcc_mean'])}")
    if 'tempo' in features:
        print(f"估计节拍: {features['tempo']:.1f} BPM")
    
    # 音频质量分析
    if 'snr_estimate' in features:
        print(f"信噪比估计: {features['snr_estimate']:.2f} dB")


def demo_multimodal_fusion() -> None:
    """演示多模态融合功能"""
    print("\n=== 多模态融合演示 ===")
    
    # 创建示例特征
    text_features = np.random.randn(128)  # 文本特征
    image_features = np.random.randn(2048)  # 图像特征  
    audio_features = np.random.randn(256)  # 音频特征
    
    print(f"文本特征维度: {text_features.shape}")
    print(f"图像特征维度: {image_features.shape}")
    print(f"音频特征维度: {audio_features.shape}")
    
    # 测试不同的融合方法
    fusion_methods = ["concatenation", "attention", "weighted", "pca"]
    
    for method in fusion_methods:
        print(f"\n--- {method.upper()} 融合 ---")
        fusion = MultimodalFusion(fusion_method=method)
        
        if method == "weighted":
            weights = [0.4, 0.4, 0.2]  # 文本、图像、音频的权重
            result = fusion.fuse_modalities(
                text_features=text_features,
                image_features=image_features,
                audio_features=audio_features,
                weights=weights
            )
        else:
            result = fusion.fuse_modalities(
                text_features=text_features,
                image_features=image_features,
                audio_features=audio_features
            )
        
        if result.get("success"):
            print(f"融合成功！输出形状: {result['output_shape']}")
        else:
            print(f"融合失败: {result.get('error', '未知错误')}")
    
    # 模态贡献度分析
    print("\n--- 模态贡献度分析 ---")
    fusion = MultimodalFusion()
    features_list = [text_features, image_features, audio_features]
    modality_names = ["text", "image", "audio"]
    
    contribution = fusion.analyze_modality_contribution(features_list, modality_names)
    
    for modality, stats in contribution.items():
        if isinstance(stats, dict) and "relative_importance" in stats:
            print(f"{modality}: 相对重要性 = {stats['relative_importance']:.3f}")


def demo_similarity_analysis() -> None:
    """演示相似度分析功能"""
    print("\n=== 相似度分析演示 ===")
    
    fusion = MultimodalFusion()
    
    # 创建两组特征用于比较
    features_a = np.random.randn(100)
    features_b = np.random.randn(100)
    features_c = features_a + np.random.randn(100) * 0.1  # 与A相似的特征
    
    # 计算不同相似度
    similarity_methods = ["cosine", "euclidean", "pearson"]
    
    for method in similarity_methods:
        sim_ab = fusion.compute_similarity(features_a, features_b, method)
        sim_ac = fusion.compute_similarity(features_a, features_c, method)
        
        print(f"{method.upper()} 相似度:")
        print(f"  A vs B: {sim_ab:.4f}")
        print(f"  A vs C: {sim_ac:.4f}")


def create_sample_data() -> None:
    """创建示例数据文件"""
    print("\n=== 创建示例数据 ===")
    
    # 创建数据目录
    data_dir = "sample_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建示例文本文件
    texts = [
        "这是第一个示例文本，包含中文内容。",
        "This is the second sample text in English.",
        "第三个文本混合了中文和English content。"
    ]
    
    with open(os.path.join(data_dir, "sample_texts.txt"), "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(f"Text {i+1}: {text}\n")
    
    print(f"示例数据已创建在 {data_dir} 目录中")


def main() -> None:
    """主函数，运行所有演示"""
    print("🚀 多模态机器学习项目演示")
    print("=" * 50)
    
    try:
        # 文本处理演示
        demo_text_processing()
        
        # 图像处理演示
        demo_image_processing()
        
        # 音频处理演示
        demo_audio_processing()
        
        # 多模态融合演示
        demo_multimodal_fusion()
        
        # 相似度分析演示
        demo_similarity_analysis()
        
        # 创建示例数据
        create_sample_data()
        
        print("\n🎉 所有演示完成！")
        print("\n💡 提示:")
        print("- 安装依赖: pip install -r requirements.txt")
        print("- 运行演示: python examples/demo.py")
        print("- 查看文档: 请参考README.md")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请确保已安装所有依赖包")


if __name__ == "__main__":
    main()
