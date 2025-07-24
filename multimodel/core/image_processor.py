"""
图像处理器模块

提供图像预处理、特征提取和图像理解功能
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from torchvision import models
import torch.nn as nn


class ImageProcessor:
    """图像处理器类，负责图像的预处理和特征提取"""
    
    def __init__(self, model_name: str = "resnet50", use_pretrained: bool = True):
        """
        初始化图像处理器
        
        Args:
            model_name: 预训练模型名称
            use_pretrained: 是否使用预训练权重
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self._load_model(use_pretrained)
        self._setup_transforms()
        
    def _load_model(self, use_pretrained: bool) -> None:
        """加载预训练模型"""
        try:
            if self.model_name == "resnet50":
                model = models.resnet50(pretrained=use_pretrained)
                # 移除最后的分类层，用于特征提取
                self.model = nn.Sequential(*list(model.children())[:-1])
            elif self.model_name == "vgg16":
                model = models.vgg16(pretrained=use_pretrained)
                model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
                self.model = model
            else:
                model = models.resnet50(pretrained=use_pretrained)
                self.model = nn.Sequential(*list(model.children())[:-1])
                
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                print(f"成功加载 {self.model_name} 模型")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
    
    def _setup_transforms(self) -> None:
        """设置图像预处理变换"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            图像数组，如果加载失败则返回None
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法加载图像: {image_path}")
                return None
            # 转换为RGB格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"图像加载错误: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        预处理图像
        
        Args:
            image: 输入图像数组
            
        Returns:
            预处理结果字典
        """
        result: Dict[str, Any] = {}
        
        # 基本信息
        result["original_shape"] = image.shape
        result["channels"] = int(image.shape[2] if len(image.shape) == 3 else 1)
        
        # 调整大小
        resized = cv2.resize(image, (224, 224))
        result["resized"] = resized
        
        # 归一化
        normalized = resized.astype(np.float32) / 255.0
        result["normalized"] = normalized
        
        # 灰度转换
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            result["grayscale"] = gray
        
        return result
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取图像特征向量
        
        Args:
            image: 输入图像数组
            
        Returns:
            图像特征向量，如果模型未加载则返回None
        """
        if self.model is None:
            print("模型未加载，无法提取特征")
            return None
            
        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(image)
            
            # 应用预处理变换
            if self.transform is not None:
                tensor_output = self.transform(pil_image)
                input_tensor = tensor_output.unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
            else:
                # 如果transform未初始化，使用默认转换
                import torchvision.transforms as transforms
                default_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                tensor_output = default_transform(pil_image)
                input_tensor = tensor_output.unsqueeze(0)
                input_tensor = input_tensor.to(self.device)
            
            # 提取特征
            if self.model is not None:
                with torch.no_grad():
                    features = self.model(input_tensor)
            else:
                # 如果模型未加载，返回随机特征
                features = torch.randn(1, 2048).to(self.device)
                features = features.squeeze().cpu().numpy()
                
            return features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        简单的对象检测（基于轮廓）
        
        Args:
            image: 输入图像数组
            
        Returns:
            检测到的对象列表
        """
        objects = []
        
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                if area > 100:  # 过滤小轮廓
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    objects.append({
                        "id": i,
                        "bbox": [x, y, w, h],
                        "area": area,
                        "center": [x + w//2, y + h//2]
                    })
            
        except Exception as e:
            print(f"对象检测失败: {e}")
        
        return objects
    
    def analyze_color(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像颜色特征
        
        Args:
            image: 输入图像数组
            
        Returns:
            颜色分析结果
        """
        result = {}
        
        try:
            # 计算每个通道的均值和标准差
            if len(image.shape) == 3:
                result["mean_rgb"] = np.mean(image, axis=(0, 1)).tolist()
                result["std_rgb"] = np.std(image, axis=(0, 1)).tolist()
                
                # 计算直方图
                hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
                hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
                
                result["histogram"] = {
                    "red": hist_r.flatten().tolist(),
                    "green": hist_g.flatten().tolist(),
                    "blue": hist_b.flatten().tolist()
                }
            
            # 主要颜色分析
            pixels = image.reshape(-1, image.shape[-1])
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            
            # 获取前5个主要颜色
            top_indices = np.argsort(counts)[-5:][::-1]
            result["dominant_colors"] = [
                {
                    "color": unique_colors[i].tolist(),
                    "count": int(counts[i]),
                    "percentage": float(counts[i] / len(pixels) * 100)
                }
                for i in top_indices
            ]
            
        except Exception as e:
            print(f"颜色分析失败: {e}")
        
        return result
    
    def batch_process(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            批量处理结果
        """
        results: Dict[str, List[Any]] = {
            "processed_images": [],
            "features": [],
            "objects": [],
            "color_analysis": []
        }
        
        for path in image_paths:
            image = self.load_image(path)
            if image is not None:
                # 预处理
                processed = self.preprocess_image(image)
                results["processed_images"].append(processed)
                
                # 特征提取
                features = self.extract_features(image)
                results["features"].append(features)
                
                # 对象检测
                objects = self.detect_objects(image)
                results["objects"].append(objects)
                
                # 颜色分析
                colors = self.analyze_color(image)
                results["color_analysis"].append(colors)
        
        return results
