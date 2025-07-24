"""
文本处理模块

提供文本预处理、特征提取和文本理解功能
"""

import re
import jieba
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TextProcessor:
    """文本处理器类，负责文本的预处理和特征提取"""
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        """
        初始化文本处理器
        
        Args:
            model_name: 预训练模型名称，默认使用中文BERT
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # 检测设备，优先使用Apple Silicon的MPS
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("使用Apple Silicon GPU (MPS)加速")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("使用CUDA GPU加速")
        else:
            self.device = torch.device("cpu")
            print("使用CPU运算")
        self._load_model()
        
    def _load_model(self):
        """加载预训练模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)  # 移动到指定设备
            self.model.eval()
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用基础文本处理功能")
    
    def clean_text(self, text: str) -> str:
        """
        清理文本数据
        
        Args:
            text: 输入文本
            
        Returns:
            清理后的文本
        """
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 转换为小写
        text = text.lower().strip()
        return text
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        return list(jieba.cut(text))
    
    def extract_features(self, text: str, max_length: int = 512) -> Optional[np.ndarray]:
        """
        提取文本特征向量
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            
        Returns:
            文本特征向量，如果模型未加载则返回None
        """
        if self.tokenizer is None or self.model is None:
            print("模型未加载，无法提取特征")
            return None
            
        try:
            # 清理文本
            clean_text = self.clean_text(text)
            
            # 编码文本
            inputs = self.tokenizer(
                clean_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # 移动输入到指定设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取特征
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用CLS token的输出作为文本表示
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            return features.squeeze()
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def batch_process(self, texts: List[str]) -> Dict[str, Any]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            处理结果字典
        """
        results: Dict[str, List[Any]] = {
            "cleaned_texts": [],
            "tokenized_texts": [],
            "features": []
        }
        
        for text in texts:
            # 清理文本
            cleaned = self.clean_text(text)
            results["cleaned_texts"].append(cleaned)
            
            # 分词
            tokens = self.tokenize_chinese(cleaned)
            results["tokenized_texts"].append(tokens)
            
            # 提取特征
            features = self.extract_features(text)
            results["features"].append(features)
        
        return results
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        获取文本统计信息
        
        Args:
            text: 输入文本
            
        Returns:
            文本统计信息字典
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_chinese(cleaned_text)
        
        return {
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "word_count": len(tokens),
            "unique_words": len(set(tokens)),
            "avg_word_length": np.mean([len(word) for word in tokens]) if tokens else 0
        }
