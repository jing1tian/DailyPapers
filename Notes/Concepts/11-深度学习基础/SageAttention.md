---
type: concept
aliases: [SageAttention, Sage Attention]
---

# SageAttention

## 定义

一种优化的注意力计算后端，通过改进内存访问模式和低精度执行，在视频扩散变换器中加速 Self-Attention 计算，可作为 PyTorch 标准注意力的 drop-in 替换。

## 核心要点

1. **即插即用**：作为标准注意力机制的替代后端，无需改变模型结构
2. **内存效率**：优化内存访问模式，减少 HBM 带宽消耗
3. **低精度加速**：在 BFloat16/FP16 下高效执行，不牺牲数值精度
4. **视频模型友好**：在长序列视频 DiT 中优势明显，历史 token 更多时收益更大

## 使用场景

适用于以下场景：
- 视频扩散变换器推理加速
- 长历史上下文的自回归世界模型
- 需要大上下文窗口的生成任务

## 代表工作

- [[GigaWorld1]]: 用于加速分层历史注意力模块，在相同 GPU 预算下支持更大历史记忆和更密集控制 token

## 相关概念

- [[注意力机制]]: SageAttention 是标准注意力的加速实现
- [[Video Diffusion Transformer]]: 主要应用场景
- [[Hierarchical Memory]]: 配合使用时收益更大
