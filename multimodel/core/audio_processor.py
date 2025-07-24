"""
音频处理器模块

提供音频预处理、特征提取和音频分析功能
"""

import librosa
import numpy as np
import soundfile as sf
import torch
from typing import List, Dict, Any, Optional, Tuple
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt


class AudioProcessor:
    """音频处理器类，负责音频的预处理和特征提取"""
    
    def __init__(self, sample_rate: int = 22050):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率，默认22050Hz
        """
        self.sample_rate = sample_rate
        self.default_duration = 30  # 默认音频长度（秒）
        
    def load_audio(self, audio_path: str, duration: Optional[float] = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            duration: 加载时长（秒），None表示加载全部
            
        Returns:
            (音频数据, 采样率) 元组，如果加载失败则返回None
        """
        try:
            audio_data, sr = librosa.load(
                audio_path, 
                sr=self.sample_rate,
                duration=duration
            )
            return audio_data, int(sr)
        except Exception as e:
            print(f"音频加载失败: {e}")
            return None
    
    def preprocess_audio(self, audio_data: np.ndarray, target_length: Optional[int] = None) -> Dict[str, Any]:
        """
        预处理音频数据
        
        Args:
            audio_data: 输入音频数据
            target_length: 目标长度（样本数），None表示不调整
            
        Returns:
            预处理结果字典
        
        Raises:
            ValueError: 输入音频数据为空或无效
        """
        result: Dict[str, Any] = {}
        
        # 基本信息
        result["original_length"] = int(audio_data.shape[0])
        result["duration"] = float(audio_data.shape[0] / self.sample_rate)
        
        # 归一化
        normalized: np.ndarray = librosa.util.normalize(audio_data)
        result["normalized"] = normalized
        
        # 调整长度
        if target_length is not None:
            if len(audio_data) > target_length:
                # 截断
                processed = audio_data[:target_length]
            else:
                # 填充
                processed = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
            result["length_adjusted"] = processed
        else:
            result["length_adjusted"] = normalized
        
        # 移除静音
        trimmed, _ = librosa.effects.trim(normalized, top_db=20)
        result["trimmed"] = trimmed
        
        return result
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        提取梅尔频谱图
        
        Args:
            audio_data: 输入音频数据
            n_mels: 梅尔滤波器数量
            
        Returns:
            梅尔频谱图
        """
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=n_mels
            )
            # 转换为对数刻度
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec
        except Exception as e:
            print(f"梅尔频谱图提取失败: {e}")
            return np.array([])
    
    def extract_mfcc(self, audio_data: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio_data: 输入音频数据
            n_mfcc: MFCC系数数量
            
        Returns:
            MFCC特征矩阵
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=n_mfcc
            )
            return mfcc
        except Exception as e:
            print(f"MFCC提取失败: {e}")
            return np.array([])
    
    def extract_chroma(self, audio_data: np.ndarray) -> np.ndarray:
        """
        提取色度特征
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            色度特征矩阵
        """
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=self.sample_rate
            )
            return chroma
        except Exception as e:
            print(f"色度特征提取失败: {e}")
            return np.array([])
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取频谱特征
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            频谱特征字典
        """
        features = {}
        
        try:
            # 频谱质心
            features["spectral_centroid"] = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0].tolist()
            
            # 频谱带宽
            features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # 频谱对比度
            features["spectral_contrast"] = librosa.feature.spectral_contrast(
                y=audio_data, sr=self.sample_rate
            )
            
            # 频谱滚降
            features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate
            )[0]
            
            # 零交叉率
            features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(audio_data)[0]
            
        except Exception as e:
            print(f"频谱特征提取失败: {e}")
        
        return features
    
    def extract_rhythm_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        提取节奏特征
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            节奏特征字典
        """
        features: Dict[str, Any] = {}
        
        try:
            # 节拍跟踪
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, 
                sr=self.sample_rate
            )
            features["tempo"] = float(tempo)
            features["beats"] = beats.tolist()
            
            # 节拍强度
            onset_strength = librosa.onset.onset_strength(
                y=audio_data, 
                sr=self.sample_rate
            )
            features["onset_strength"] = onset_strength.tolist()  # type: ignore
            
        except Exception as e:
            print(f"节奏特征提取失败: {e}")
        
        return features
    
    def analyze_audio_quality(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        分析音频质量
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            音频质量分析结果
        """
        quality = {}
        
        try:
            # 信噪比估计
            # 简单估计：假设最小值为噪音，最大值为信号
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.mean((audio_data - np.mean(audio_data)) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            quality["snr_estimate"] = float(snr)
            
            # 动态范围
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            quality["dynamic_range"] = float(dynamic_range)
            
            # 音频能量
            energy = np.sum(audio_data ** 2)
            quality["energy"] = float(energy)
            
            # RMS能量
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            quality["rms_energy"] = rms_energy.tolist()
            
            # 静音检测
            silence_threshold = 0.01
            silent_frames = np.sum(np.abs(audio_data) < silence_threshold)
            silence_ratio = silent_frames / len(audio_data)
            quality["silence_ratio"] = float(silence_ratio)
            
        except Exception as e:
            print(f"音频质量分析失败: {e}")
        
        return quality
    
    def extract_comprehensive_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        提取综合音频特征
        
        Args:
            audio_data: 输入音频数据
            
        Returns:
            综合特征字典
        """
        features = {}
        
        # MFCC特征
        mfcc = self.extract_mfcc(audio_data)
        if mfcc.size > 0:
            features["mfcc"] = mfcc
            features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
            features["mfcc_std"] = np.std(mfcc, axis=1).tolist()
        
        # 梅尔频谱图
        mel_spec = self.extract_mel_spectrogram(audio_data)
        if mel_spec.size > 0:
            features["mel_spectrogram"] = mel_spec
            features["mel_mean"] = np.mean(mel_spec, axis=1).tolist()
        
        # 色度特征
        chroma = self.extract_chroma(audio_data)
        if chroma.size > 0:
            features["chroma"] = chroma
            features["chroma_mean"] = np.mean(chroma, axis=1).tolist()
        
        # 频谱特征
        spectral_features = self.extract_spectral_features(audio_data)
        for key, value in spectral_features.items():
            features[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        # 节奏特征
        rhythm_features = self.extract_rhythm_features(audio_data)
        features.update(rhythm_features)
        
        # 音频质量
        quality_features = self.analyze_audio_quality(audio_data)
        features.update(quality_features)
        
        return features
    
    def batch_process(self, audio_paths: List[str]) -> Dict[str, Any]:
        """
        批量处理音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            批量处理结果
        """
        results: Dict[str, List[Any]] = {
            "processed_audio": [],
            "features": [],
            "file_info": []
        }
        
        for path in audio_paths:
            audio_result = self.load_audio(path)
            if audio_result is not None:
                audio_data, sr = audio_result
                
                # 文件信息
                file_info = {
                    "path": path,
                    "sample_rate": sr,
                    "duration": len(audio_data) / sr,
                    "samples": len(audio_data)
                }
                results["file_info"].append(file_info)
                
                # 预处理
                processed = self.preprocess_audio(audio_data)
                results["processed_audio"].append(processed)
                
                # 特征提取
                features = self.extract_comprehensive_features(audio_data)
                results["features"].append(features)
        
        return results
