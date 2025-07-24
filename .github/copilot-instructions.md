<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# MultiModel 项目 Copilot 指令

这是一个多模态机器学习项目，专注于文本、图像和音频数据的处理与融合。

## 项目概述

- **项目类型**: Python 多模态机器学习框架
- **主要功能**: 文本处理、图像处理、音频处理、多模态融合
- **技术栈**: PyTorch, Transformers, OpenCV, Librosa, scikit-learn

## 代码规范

### Python 代码风格
- 遵循 PEP 8 规范
- 使用类型提示 (Type Hints)
- 函数和类需要完整的文档字符串
- 错误处理要充分，避免程序崩溃

### 文档字符串格式
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    函数简要描述
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
        
    Returns:
        返回值描述
        
    Raises:
        ExceptionType: 异常情况描述
    """
```

### 模块结构
- `multimodel/core/`: 核心处理模块
- `multimodel/utils/`: 工具函数
- `examples/`: 示例代码
- `tests/`: 测试代码

## 技术要求

### 深度学习框架
- 优先使用 PyTorch 2.0+
- 模型加载时要处理异常情况
- 支持 GPU/CPU 自动切换
- 使用预训练模型时要考虑内存和性能

### 数据处理
- 支持批处理操作
- 异常数据要有适当的错误处理
- 大文件要考虑内存使用
- 支持多种数据格式

### 多模态融合
- 特征维度要对齐
- 支持多种融合策略
- 要有特征标准化
- 考虑不同模态的权重

## 编程最佳实践

1. **错误处理**: 使用 try-catch 块，提供有意义的错误信息
2. **性能优化**: 避免不必要的计算，使用向量化操作
3. **内存管理**: 及时释放大型对象，使用生成器处理大数据
4. **可扩展性**: 设计时考虑新模态和算法的添加
5. **测试友好**: 代码要易于单元测试

## 常用库和模式

### 导入模式
```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
```

### 异常处理模式
```python
try:
    # 处理逻辑
    pass
except SpecificException as e:
    print(f"具体错误信息: {e}")
    return None  # 或适当的默认值
```

### 配置模式
- 使用类属性设置默认参数
- 支持运行时配置修改
- 配置验证和类型检查

## AI 辅助开发指导

当使用 GitHub Copilot 时：

1. **优先考虑**: 代码的可读性和维护性
2. **注意**: 机器学习相关的最佳实践
3. **确保**: 适当的错误处理和边界条件检查
4. **包含**: 完整的类型提示和文档字符串
5. **考虑**: 性能影响和内存使用

## 具体模块指导

### TextProcessor
- 支持中英文混合处理
- 使用 Transformers 库
- 考虑分词的准确性
- 特征提取要高效

### ImageProcessor  
- 使用 OpenCV 和 PIL
- 支持多种图像格式
- 预处理要标准化
- 考虑图像增强

### AudioProcessor
- 使用 Librosa 处理音频
- 支持多种音频特征
- 处理不同采样率
- 考虑音频质量

### MultimodalFusion
- 特征对齐和标准化
- 支持多种融合策略
- 权重学习和优化
- 结果可解释性
