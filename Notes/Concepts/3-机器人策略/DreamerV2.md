---
type: concept
aliases: [Dreamer V2, Dreamer2]
---

# DreamerV2

## 定义
DreamerV2 是基于潜在动力学的世界模型，使用 RSSM（Recurrent State Space Model）在离散潜在空间中学习环境动力学，并在 WM 内部的想象轨迹上训练 actor-critic。

## 数学形式
RSSM 的状态转移：
$$h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), \quad z_t \sim q_\phi(z_t | h_t, x_t)$$

## 核心要点
1. **离散潜在空间**：使用直通估计器（straight-through estimator）训练离散 token
2. **想象中训练**：在世界模型 rollout 的"想象"中训练 actor-critic
3. **相比 V1 改进**：离散表示、KL balancing、更稳健的训练

## 代表工作
- DreamerV2 原始论文（Ha & Schmidhuber 系列）
- [[DreamerV3]]: 更通用的改进版本
- [[EA-WM]]: 将 DreamerV2 作为对比基线

## 相关概念
- [[DreamerV3]]
- [[RSSM]]
- [[JEPA]]
