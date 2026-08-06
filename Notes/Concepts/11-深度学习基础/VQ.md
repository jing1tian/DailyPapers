---
type: concept
aliases: [Vector Quantization, 向量量化]
---

# VQ

## 定义
Vector Quantization：将连续特征向量映射到离散 codebook 中最近邻码字的量化方法，是 VQ-VAE 等模型的核心组件。

## 数学形式
$$z_q = \arg\min_{e_k \in E} \|z_e - e_k\|_2$$

## 核心要点
1. 学习离散的 codebook（码本）$E = \{e_1, ..., e_K\}$
2. straight-through estimator 处理不可微的 argmin
3. 离散表征有利于自回归建模

## 代表工作
- [[DPA-IL]]: 用 VQ 量化高层行为选择的离散动作空间

## 相关概念
- [[MAGVIT]]
- [[EMU-3]]
