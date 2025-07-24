"""
多模态融合模块

提供多种模态数据的融合和联合学习功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json


class MultimodalFusion:
    """多模态融合器类，负责不同模态数据的融合"""
    
    def __init__(self):
        """初始化多模态融合器"""
        self.pca = None
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征
        
        Args:
            features: 输入特征矩阵
            
        Returns:
            标准化后的特征矩阵
        """
        if features.size == 0:
            return features
        
        # 确保features是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 对于每个特征矩阵，都创建独立的scaler
        # 这样避免了不同维度特征之间的冲突
        local_scaler = StandardScaler()
        normalized = local_scaler.fit_transform(features)
        
        return normalized
    
    def concatenation_fusion(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        简单连接融合
        
        Args:
            features_list: 特征列表
            
        Returns:
            融合后的特征向量
        """
        # 确保所有特征都是2D数组
        processed_features = []
        for features in features_list:
            if features is not None and features.size > 0:
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                elif features.ndim > 2:
                    features = features.reshape(features.shape[0], -1)
                processed_features.append(features)
        
        if not processed_features:
            return np.array([])
        
        # 连接所有特征
        try:
            fused_features = np.concatenate(processed_features, axis=1)
            return fused_features
        except Exception as e:
            print(f"特征连接失败: {e}")
            return np.array([])
    
    def weighted_fusion(self, features_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        加权融合
        
        Args:
            features_list: 特征列表
            weights: 权重列表
            
        Returns:
            加权融合后的特征向量
        """
        if len(features_list) != len(weights):
            raise ValueError("特征列表和权重列表长度不匹配")
        
        # 归一化权重
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)
        
        # 简单的特征标准化（对每个特征向量单独标准化）
        normalized_features = []
        for features in features_list:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # 对每个特征向量进行零均值单位方差标准化
            mean = np.mean(features, axis=1, keepdims=True)
            std = np.std(features, axis=1, keepdims=True)
            std = np.where(std == 0, 1, std)  # 避免除零
            normalized = (features - mean) / std
            normalized_features.append(normalized)
        
        # 加权求和
        weighted_features = []
        for i, features in enumerate(normalized_features):
            weight = weights_array[i]
            weighted_features.append(features * weight)
        
        if not weighted_features:
            return np.array([])
        
        # 如果特征维度不同，先进行维度对齐
        max_dim = max(f.shape[1] for f in weighted_features)
        aligned_features = []
        
        for features in weighted_features:
            if features.shape[1] < max_dim:
                # 零填充
                padding = np.zeros((features.shape[0], max_dim - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            elif features.shape[1] > max_dim:
                # 截断
                features = features[:, :max_dim]
            aligned_features.append(features)
        
        # 求和
        fused_features = np.sum(aligned_features, axis=0)
        return fused_features
    
    def attention_fusion(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        注意力机制融合
        
        Args:
            features_list: 特征列表
            
        Returns:
            注意力融合后的特征向量
        """
        # 简化的注意力机制实现
        valid_features = [f for f in features_list if f is not None and f.size > 0]
        
        if not valid_features:
            return np.array([])
        
        # 计算每个特征的重要性分数
        importance_scores = []
        for features in valid_features:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            # 使用方差作为重要性度量
            importance = np.var(features, axis=1).mean()
            importance_scores.append(importance)
        
        # 归一化重要性分数作为注意力权重
        attention_weights = np.array(importance_scores)
        attention_weights = attention_weights / (np.sum(attention_weights) + 1e-10)
        
        # 应用注意力权重
        return self.weighted_fusion(valid_features, attention_weights.tolist())
    
    def pca_fusion(self, features_list: List[np.ndarray], n_components: int = 100) -> np.ndarray:
        """
        PCA降维融合
        
        Args:
            features_list: 特征列表
            n_components: PCA组件数量
            
        Returns:
            PCA降维后的特征向量
        """
        # 先进行连接融合
        concatenated = self.concatenation_fusion(features_list)
        
        if concatenated.size == 0:
            return np.array([])
        
        # 调整PCA组件数量
        actual_components = min(n_components, concatenated.shape[1], concatenated.shape[0])
        
        if self.pca is None:
            self.pca = PCA(n_components=actual_components)
            reduced_features = self.pca.fit_transform(concatenated)
        else:
            reduced_features = self.pca.transform(concatenated)
        
        return reduced_features
    
    def fuse_modalities(self, 
                       text_features: Optional[np.ndarray] = None,
                       image_features: Optional[np.ndarray] = None,
                       audio_features: Optional[np.ndarray] = None,
                       custom_features: Optional[List[np.ndarray]] = None,
                       weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        融合多模态特征
        
        Args:
            text_features: 文本特征
            image_features: 图像特征
            audio_features: 音频特征
            custom_features: 自定义特征列表
            weights: 权重列表（用于加权融合）
            
        Returns:
            融合结果字典
        """
        # 收集所有有效特征
        features_list = []
        modality_names = []
        
        if text_features is not None and text_features.size > 0:
            features_list.append(text_features)
            modality_names.append("text")
        
        if image_features is not None and image_features.size > 0:
            features_list.append(image_features)
            modality_names.append("image")
        
        if audio_features is not None and audio_features.size > 0:
            features_list.append(audio_features)
            modality_names.append("audio")
        
        if custom_features:
            for i, features in enumerate(custom_features):
                if features is not None and features.size > 0:
                    features_list.append(features)
                    modality_names.append(f"custom_{i}")
        
        if not features_list:
            return {"error": "没有有效的特征输入"}
        
        # 根据融合方法进行融合
        result: Dict[str, Any] = {
            "modalities": modality_names,
            "fusion_method": self.fusion_method,
            "input_shapes": [f.shape for f in features_list]
        }
        
        try:
            if self.fusion_method == "concatenation":
                fused_features = self.concatenation_fusion(features_list)
            elif self.fusion_method == "weighted":
                if weights is None:
                    # 如果没有提供权重，使用平均权重
                    weights = [1.0 / len(features_list)] * len(features_list)
                fused_features = self.weighted_fusion(features_list, weights)
            elif self.fusion_method == "attention":
                fused_features = self.attention_fusion(features_list)
            elif self.fusion_method == "pca":
                fused_features = self.pca_fusion(features_list)
            else:
                # 默认使用连接融合
                fused_features = self.concatenation_fusion(features_list)
            
            result["fused_features"] = fused_features
            result["output_shape"] = fused_features.shape
            result["success"] = True
            
        except Exception as e:
            result["error"] = f"融合失败: {e}"
            result["success"] = False
        
        return result
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray, 
                          method: str = "cosine") -> float:
        """
        计算特征相似度
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            method: 相似度计算方法 ("cosine", "euclidean", "pearson")
            
        Returns:
            相似度分数
        """
        try:
            # 确保特征向量是一维的
            if features1.ndim > 1:
                features1 = features1.flatten()
            if features2.ndim > 1:
                features2 = features2.flatten()
            
            # 调整向量长度
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
            
            if method == "cosine":
                # 余弦相似度
                dot_product = np.dot(features1, features2)
                norm1 = np.linalg.norm(features1)
                norm2 = np.linalg.norm(features2)
                similarity = dot_product / (norm1 * norm2 + 1e-10)
                
            elif method == "euclidean":
                # 欧氏距离（转换为相似度）
                distance = np.linalg.norm(features1 - features2)
                similarity = 1 / (1 + distance)
                
            elif method == "pearson":
                # 皮尔逊相关系数
                correlation_matrix = np.corrcoef(features1, features2)
                similarity = correlation_matrix[0, 1]
                
            else:
                similarity = 0.0
            
            return float(similarity)
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return 0.0
    
    def analyze_modality_contribution(self, features_list: List[np.ndarray], 
                                    modality_names: List[str]) -> Dict[str, Any]:
        """
        分析各模态的贡献度
        
        Args:
            features_list: 特征列表
            modality_names: 模态名称列表
            
        Returns:
            模态贡献度分析结果
        """
        analysis: Dict[str, Any] = {}
        
        try:
            # 计算每个模态的统计信息
            for i, (features, name) in enumerate(zip(features_list, modality_names)):
                if features is not None and features.size > 0:
                    if features.ndim == 1:
                        features = features.reshape(1, -1)
                    
                    stats = {
                        "mean": float(np.mean(features)),
                        "std": float(np.std(features)),
                        "variance": float(np.var(features)),
                        "max": float(np.max(features)),
                        "min": float(np.min(features)),
                        "dimension": features.shape[1],
                        "sparsity": float(np.sum(features == 0) / features.size)
                    }
                    analysis[name] = stats
            
            # 计算相对重要性
            if len(analysis) > 1:
                variances = [stats["variance"] for stats in analysis.values()]
                total_variance = sum(variances)
                
                for name, variance in zip(analysis.keys(), variances):
                    analysis[name]["relative_importance"] = variance / (total_variance + 1e-10)
        
        except Exception as e:
            analysis["error"] = f"贡献度分析失败: {e}"
        
        return analysis
    
    def save_fusion_config(self, filepath: str):
        """
        保存融合配置
        
        Args:
            filepath: 配置文件路径
        """
        config = {
            "fusion_method": self.fusion_method,
            "is_fitted": self.is_fitted,
            "scaler_params": {
                "mean_": self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None else None,
                "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None else None
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"融合配置已保存到: {filepath}")
        except Exception as e:
            print(f"配置保存失败: {e}")
    
    def load_fusion_config(self, filepath: str):
        """
        加载融合配置
        
        Args:
            filepath: 配置文件路径
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.fusion_method = config.get("fusion_method", "concatenation")
            self.is_fitted = config.get("is_fitted", False)
            
            # 恢复scaler参数
            if config.get("scaler_params"):
                scaler_params = config["scaler_params"]
                if scaler_params.get("mean_"):
                    self.scaler.mean_ = np.array(scaler_params["mean_"])
                if scaler_params.get("scale_"):
                    self.scaler.scale_ = np.array(scaler_params["scale_"])
            
            print(f"融合配置已从 {filepath} 加载")
            
        except Exception as e:
            print(f"配置加载失败: {e}")
