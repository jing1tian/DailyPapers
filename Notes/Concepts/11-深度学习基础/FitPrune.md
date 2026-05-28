---
type: concept
aliases: [FitPrune]
---

# FitPrune

## 定义

FitPrune 是一种 VLA 推理加速的 token 剪枝方法，通过对动作预测任务做适配性（fitness）评估来选择保留的 visual token，是专门为 VLA 设计的剪枝方法之一。

## 核心要点

1. **任务适配性评分**: 用任务特定的适配性指标（而非通用语义相关性）为 token 排序
2. **VLA 专用**: 相比 FastV、SparseVLM 更针对动作预测优化
3. **评测对比**: 在 [[LIBERO]]、[[SIMPLER]] 上与 SAGPruning 等方法对比

## 相关概念

- [[FastV]]
- [[SparseVLM]]
- [[DivPrune]]
- [[EfficientVLA]]
