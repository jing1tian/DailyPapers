---
type: concept
aliases: [FastV, Fast Visual Token Pruning]
---

# FastV

## 定义

FastV 是一种用于 VLM 推理加速的视觉 token 剪枝方法，通过分析注意力权重识别重要性低的 visual token 并在中间层丢弃，在保持语义质量的同时降低推理计算量。

## 核心要点

1. **注意力驱动**: 用 transformer 中间层的 attention score 评估 visual token 重要性
2. **渐进丢弃**: 在特定层之后丢弃低注意力权重的 token，减少后续层计算
3. **零训练成本**: 推理时直接剪枝，无需重新训练模型
4. **语义-动作 Gap**: 在 VLA 中语义重要的 token 不等于动作预测重要的 token（见 [[SAGPruning]]）

## 代表工作

- [[FastV]]（原方法）: VLM 视觉 token 剪枝，CVPR 2024

## 相关概念

- [[SparseVLM]]
- [[DivPrune]]
- [[FitPrune]]
- [[OpenVLA]]
