---
type: concept
aliases: [Fast TD3, Faster Twin Delayed DDPG]
---

# FastTD3

## 定义
FastTD3：TD3（Twin Delayed DDPG）的高效变体，通过计算和训练流程优化实现更快的 wall-clock 训练速度，适用于需要大规模 offline-to-online RL 微调的 loco-manipulation 等复杂任务。

## 数学形式
TD3 actor 更新（保持不变）：
$$\nabla_\phi J(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\left[\nabla_a Q_{\theta_1}(s,a)|_{a=\mu_\phi(s)} \nabla_\phi \mu_\phi(s)\right]$$

FastTD3 主要在批次大小、目标网络更新频率、向量化环境方面做工程优化。

## 核心要点
1. 在标准 [[TD3]] 基础上通过批次并行、更大 replay buffer 等工程手段提速
2. 适合需要大量样本的 offline-to-online 设置（如先 offline 预训练再 online 微调）
3. 被用于 ETH Zurich 的 Loco-Manipulation 工作（SMPC 演示 + 稀疏 RL）
4. 与 MuJoCo 的向量化仿真（MJX）配合使用效果最佳

## 代表工作
- 用于 Learning Loco-Manipulation From SMPC Demonstrations（2608.12063）

## 相关概念
- [[TD3]]: 基础算法（Twin Delayed DDPG）
- [[SAC]]: 另一常用连续动作 RL 算法
- [[MPC]]: FastTD3 在该工作中与 SMPC 演示生成配合
