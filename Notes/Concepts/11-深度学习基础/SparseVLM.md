---
type: concept
aliases: [SparseVLM, Sparse Visual Language Model]
---

# SparseVLM

## 定义

SparseVLM 是一种 VLM 推理加速方法，通过文本引导的稀疏化策略识别并保留与文本查询最相关的视觉 token，压缩视觉序列长度以降低计算开销。

## 核心要点

1. **文本引导稀疏化**: 用文本 query 与 visual token 的相关性评分作为保留标准
2. **无需微调**: 可直接应用于预训练 VLM 推理阶段
3. **与 VLA 的语义-动作 Gap**: 文本相关不等于动作预测相关，直接用于 VLA 效果有限

## 代表工作

- [[SparseVLM]]（原方法）: 文本引导稀疏视觉 token
- [[VLA-Pruner]]: 指出 SparseVLM 直接用于 VLA 会因语义-动作差异导致性能下降，提出双流重要性估计

## 相关概念

- [[FastV]]
- [[DivPrune]]
- [[FitPrune]]
- [[EfficientVLA]]
