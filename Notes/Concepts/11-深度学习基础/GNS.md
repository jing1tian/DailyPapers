---
type: concept
aliases: ["Graph Network Simulator"]
---

# GNS

## 定义
图神经网络物理仿真器，将粒子/刚体建模为图节点，用 GNN 消息传递预测物理动力学。

## 数学形式
$$\hat{a}_i = \phi^v\left(h_i, \sum_{j \in \mathcal{N}(i)} \phi^e(h_i, h_j, e_{ij})\right)$$

## 核心要点
1. GPWM 的基础对比方法
2. GNN 在物理仿真上的里程碑工作（DeepMind）

## 代表工作
- 见关联笔记

## 相关概念
- [[GPWM]]
- [[ParticleWorlds]]
