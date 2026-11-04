---
type: concept
aliases: [Policy-Initialized Latent Planning]
---

# PILOT（Policy-Initialized Latent Planning）

## 定义
从已有策略出发初始化规划搜索的世界模型规划方法，利用 policy rollout 提供初始轨迹分布，降低 planning 的搜索成本。

## 数学形式
$$\tau^* = \arg\max_\tau \sum_t r(s_t, a_t), \quad \tau_0 \sim \pi_\theta(a | s)$$

用策略 $\pi_\theta$ 生成初始轨迹候选，再用 WM 评分并细化。

## 核心要点
1. 相比从随机噪声初始化的 MPC，policy 给出的初始化更接近优质解，搜索效率高
2. 结合 WM 评分函数可在 inference 时做进一步优化
3. 是 pico-WAM 中对比的规划策略之一

## 代表工作
- [[pico-WAM]]: 在 Push-T 上对比 PILOT 与其他规划方法
- [[DINO-WM]]: 相似的 latent planning 思路

## 相关概念
- [[DINO-WM]]
- [[MCTS]]
- [[PETS]]
