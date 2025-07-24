"""
命令行接口模块

提供命令行工具来使用多模态处理功能
"""

import click
import os
import json
from pathlib import Path
from multimodel import TextProcessor, ImageProcessor, AudioProcessor, MultimodalFusion


@click.group()
@click.version_option(version="0.1.0")
def main():
    """多模态机器学习项目命令行工具"""
    pass


@main.command()
@click.argument('text')
@click.option('--output', '-o', help='输出文件路径')
@click.option('--model', '-m', default='bert-base-chinese', help='使用的预训练模型')
def process_text(text, output, model):
    """处理文本数据"""
    click.echo(f"使用模型 {model} 处理文本...")
    
    processor = TextProcessor(model_name=model)
    
    # 处理文本
    result = {
        'original_text': text,
        'cleaned_text': processor.clean_text(text),
        'tokens': processor.tokenize_chinese(text),
        'statistics': processor.get_text_statistics(text)
    }
    
    # 提取特征
    features = processor.extract_features(text)
    if features is not None:
        result['features_shape'] = features.shape
        result['features_mean'] = float(features.mean())
        result['features_std'] = float(features.std())
    
    # 输出结果
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"结果已保存到: {output}")
    else:
        click.echo("处理结果:")
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@main.command()
@click.argument('image_path')
@click.option('--output', '-o', help='输出目录')
@click.option('--model', '-m', default='resnet50', help='使用的预训练模型')
def process_image(image_path, output, model):
    """处理图像数据"""
    if not os.path.exists(image_path):
        click.echo(f"错误: 图像文件不存在: {image_path}")
        return
    
    click.echo(f"使用模型 {model} 处理图像: {image_path}")
    
    processor = ImageProcessor(model_name=model)
    
    # 加载图像
    image = processor.load_image(image_path)
    if image is None:
        click.echo("错误: 无法加载图像")
        return
    
    # 处理图像
    processed = processor.preprocess_image(image)
    objects = processor.detect_objects(image)
    colors = processor.analyze_color(image)
    
    result = {
        'image_path': image_path,
        'original_shape': processed['original_shape'],
        'detected_objects': len(objects),
        'dominant_colors': len(colors.get('dominant_colors', [])),
        'objects_details': objects[:5],  # 只保存前5个对象
        'color_info': {
            'mean_rgb': colors.get('mean_rgb', []),
            'dominant_colors': colors.get('dominant_colors', [])[:3]  # 前3个主要颜色
        }
    }
    
    # 提取特征
    features = processor.extract_features(image)
    if features is not None:
        result['features_shape'] = features.shape
        result['features_mean'] = float(features.mean())
        result['features_std'] = float(features.std())
    
    # 输出结果
    if output:
        os.makedirs(output, exist_ok=True)
        output_file = os.path.join(output, f"{Path(image_path).stem}_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"结果已保存到: {output_file}")
    else:
        click.echo("处理结果:")
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@main.command()
@click.argument('audio_path')
@click.option('--output', '-o', help='输出目录')
@click.option('--duration', '-d', type=float, help='处理时长（秒）')
def process_audio(audio_path, output, duration):
    """处理音频数据"""
    if not os.path.exists(audio_path):
        click.echo(f"错误: 音频文件不存在: {audio_path}")
        return
    
    click.echo(f"处理音频: {audio_path}")
    
    processor = AudioProcessor()
    
    # 加载音频
    audio_result = processor.load_audio(audio_path, duration=duration)
    if audio_result is None:
        click.echo("错误: 无法加载音频")
        return
    
    audio_data, sr = audio_result
    
    # 处理音频
    processed = processor.preprocess_audio(audio_data)
    features = processor.extract_comprehensive_features(audio_data)
    
    result = {
        'audio_path': audio_path,
        'sample_rate': sr,
        'duration': processed['duration'],
        'original_length': processed['original_length']
    }
    
    # 添加特征信息
    if 'tempo' in features:
        result['tempo'] = features['tempo']
    if 'mfcc_mean' in features:
        result['mfcc_features'] = len(features['mfcc_mean'])
    if 'snr_estimate' in features:
        result['snr_estimate'] = features['snr_estimate']
    if 'silence_ratio' in features:
        result['silence_ratio'] = features['silence_ratio']
    
    # 输出结果
    if output:
        os.makedirs(output, exist_ok=True)
        output_file = os.path.join(output, f"{Path(audio_path).stem}_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        click.echo(f"结果已保存到: {output_file}")
    else:
        click.echo("处理结果:")
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))


@main.command()
@click.option('--text', help='文本内容或文件路径')
@click.option('--image', help='图像文件路径')
@click.option('--audio', help='音频文件路径')
@click.option('--method', '-m', default='concatenation', 
              type=click.Choice(['concatenation', 'attention', 'weighted', 'pca']),
              help='融合方法')
@click.option('--weights', help='权重列表，用逗号分隔（仅用于weighted方法）')
@click.option('--output', '-o', help='输出文件路径')
def fuse(text, image, audio, method, weights, output):
    """融合多模态数据"""
    
    if not any([text, image, audio]):
        click.echo("错误: 至少需要提供一种模态的数据")
        return
    
    click.echo(f"使用 {method} 方法进行多模态融合...")
    
    # 初始化处理器
    features = {}
    
    # 处理文本
    if text:
        text_processor = TextProcessor()
        if os.path.exists(text):
            with open(text, 'r', encoding='utf-8') as f:
                text_content = f.read()
        else:
            text_content = text
        
        text_features = text_processor.extract_features(text_content)
        if text_features is not None:
            features['text'] = text_features
            click.echo(f"文本特征维度: {text_features.shape}")
    
    # 处理图像
    if image and os.path.exists(image):
        image_processor = ImageProcessor()
        img_data = image_processor.load_image(image)
        if img_data is not None:
            img_features = image_processor.extract_features(img_data)
            if img_features is not None:
                features['image'] = img_features
                click.echo(f"图像特征维度: {img_features.shape}")
    
    # 处理音频
    if audio and os.path.exists(audio):
        audio_processor = AudioProcessor()
        audio_result = audio_processor.load_audio(audio)
        if audio_result is not None:
            audio_data, _ = audio_result
            audio_features = audio_processor.extract_comprehensive_features(audio_data)
            # 使用MFCC特征作为音频表示
            if 'mfcc' in audio_features:
                mfcc = audio_features['mfcc']
                audio_feat = mfcc.mean(axis=1)  # 对时间维度求平均
                features['audio'] = audio_feat
                click.echo(f"音频特征维度: {audio_feat.shape}")
    
    if not features:
        click.echo("错误: 无法提取任何有效特征")
        return
    
    # 进行融合
    fusion = MultimodalFusion(fusion_method=method)
    
    # 准备权重
    fusion_weights = None
    if method == 'weighted' and weights:
        try:
            fusion_weights = [float(w.strip()) for w in weights.split(',')]
            if len(fusion_weights) != len(features):
                click.echo(f"警告: 权重数量({len(fusion_weights)})与模态数量({len(features)})不匹配")
                fusion_weights = None
        except ValueError:
            click.echo("警告: 权重格式错误，使用默认融合方法")
            fusion_weights = None
    
    # 执行融合
    result = fusion.fuse_modalities(
        text_features=features.get('text'),
        image_features=features.get('image'),
        audio_features=features.get('audio'),
        weights=fusion_weights
    )
    
    if result.get('success'):
        click.echo(f"融合成功！输出特征维度: {result['output_shape']}")
        
        # 分析模态贡献
        features_list = [features[k] for k in features.keys()]
        modality_names = list(features.keys())
        contribution = fusion.analyze_modality_contribution(features_list, modality_names)
        
        summary = {
            'fusion_method': method,
            'input_modalities': modality_names,
            'output_shape': result['output_shape'],
            'modality_contribution': contribution
        }
        
        # 输出结果
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            click.echo(f"融合结果已保存到: {output}")
        else:
            click.echo("融合结果:")
            click.echo(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        click.echo(f"融合失败: {result.get('error', '未知错误')}")


@main.command()
def demo():
    """运行演示程序"""
    click.echo("启动多模态项目演示...")
    try:
        from examples.demo import main as demo_main
        demo_main()
    except ImportError:
        click.echo("错误: 无法导入演示模块，请确保项目结构完整")
    except Exception as e:
        click.echo(f"演示运行错误: {e}")


if __name__ == '__main__':
    main()
