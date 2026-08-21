---
type: concept
aliases: [Sparse Partition-Reconstruct, SparsePR]
---

# SparsePR

## 定义
一种无训练的块稀疏注意力方法，通过 Response-Coupled Partitioning 选择稀疏子集并用 Probe-Fitted Residual Reconstruction 修复被丢弃交互的误差，用于加速视频生成 DiT 和世界模型推理。

## 核心要点
1. Response-Coupled Partitioning：基于 query-key attention 响应相关性切分稀疏块，比 naive block-sparse 更精准
2. Probe-Fitted Residual Reconstruction：用轻量 affine 模型估计并补偿丢弃 attention 的残差误差
3. 无需额外训练，可直接插入已有 DiT/世界模型推理 pipeline

## 代表工作
- 原始论文：Taghavi et al., 2026 — [[SparsePR]] (arXiv 2608.18484)

## 相关概念
- [[HunyuanVideo]]
- [[WorldPack]]
