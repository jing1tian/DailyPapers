---
type: concept
aliases: [Behavior Score Optimization]
---

# BESO

## 定义
基于 score function 的机器人操作策略方法，使用能量函数对动作分布建模，通过 score matching 训练策略并在推理时用 Langevin dynamics 采样动作。

## 核心要点
1. 类扩散模型框架，但专为机器人连续控制设计
2. 比 [[Diffusion-Policy]] 推理速度更快（更少去噪步骤）
3. [[PHIL]] 以 BESO 为基础策略架构

## 相关概念
- [[Diffusion-Policy]]
- [[PHIL]]
- [[Action Diffusion]]
