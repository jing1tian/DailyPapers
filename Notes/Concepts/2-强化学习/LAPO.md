---
type: concept
aliases: [Latent Action Policy Optimization]
---

# LAPO

## 定义
Latent Action Policy Optimization，从无动作标注视频中自监督学习潜在动作表征，然后用于策略学习的方法。

## 核心要点
1. 从纯视频（无 action label）中学习结构化潜动作空间
2. 潜动作通过预测下一帧变化来定义，具有物理语义
3. 与 [[CLAW]] 对比：LAPO 用预测损失约束潜动作，CLAW 用对抗性正则化
4. 下游可用 MPC 或 RL 进行规划

## 代表工作
- [[CLAW]]: 对比了 LAPO，提出用对抗正则化改进潜动作学习
- [[LAM]]: 相关的 Latent Action Model 方法

## 相关概念
- [[CLAW]]
- [[LAM]]
- [[LAPA]]
- [[action-conditioned world model]]
