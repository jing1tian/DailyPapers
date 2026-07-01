---
type: concept
aliases: [Behavior Transformer, BeT]
---

# BET（Behavior Transformer）

## 定义
一种基于 Transformer + VQ 离散化的模仿学习策略，将连续动作离散化为 codebook token，再用 Transformer 建模动作分布。

## 核心要点
1. 动作空间通过 k-means 聚类离散化为 codebook
2. Transformer 预测 residual 细化离散中心
3. 适合多模态动作分布（抓取等 bimodal 任务）

## 代表工作
- [[SA-VLA]]: 对比 BET 离散化方案

## 相关概念
- [[Autoregressive Policy]]
- [[FAST]]
