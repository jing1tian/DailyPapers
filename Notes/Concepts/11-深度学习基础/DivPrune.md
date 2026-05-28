---
type: concept
aliases: [DivPrune, Diversity-Based Pruning]
---

# DivPrune

## 定义

DivPrune 是一种基于多样性的视觉 token 剪枝方法，保留语义多样性最高的 token 子集，避免信息冗余，用于 VLM/VLA 推理加速。

## 核心要点

1. **多样性优先**: 选择在特征空间中分布最分散的 token，保留最大信息量
2. **与注意力方法对比**: FastV 用注意力权重、DivPrune 用特征多样性，两者标准不同
3. **VLA 局限**: 与 FastV 和 SparseVLM 同样面临语义-动作 gap 问题

## 代表工作

- [[DivPrune]]（原方法）: 多样性驱动 token 剪枝

## 相关概念

- [[FastV]]
- [[SparseVLM]]
- [[FitPrune]]
