---
type: concept
aliases: [Dynamic Vision Transformer]
---

# DynamicViT

## 定义
一种动态 token 剪枝方法，训练一个轻量 prediction module 在每层 ViT 前决定保留哪些 token，丢弃贡献低的 patch，降低 ViT 推理计算量。

## 核心要点
1. 每层自适应决定保留的 token 比例
2. 用 token 重要性预测网络 + 可微近似（Gumbel-Softmax）实现 end-to-end 训练
3. 推理时显著降低 FLOPs，精度损失小

## 代表工作
- Rao et al., DynamicViT, NeurIPS 2021

## 相关概念
- [[EViT]]
- [[APS-VLA]]
- [[Token Pruning]]
