---
type: concept
aliases: [Advantage-Weighted Regression, 优势加权回归]
---

# AWR（Advantage-Weighted Regression）

## 定义

AWR 是一种 off-policy 强化学习算法，通过将行为克隆损失与优势加权相结合，利用历史数据中的高优势样本来优化策略，无需显式的策略梯度估计。

## 数学形式

$$
\mathcal{L}_{\text{AWR}} = \mathbb{E}_{t}\!\left[-\log \pi(a_{t}|o_{t}) \exp\!\left(\frac{1}{\beta}(G_t - V(o_t))\right)\right]
$$

其中优势估计 $A_t = G_t - V(o_t)$ 作为 BC 损失的指数权重，$\beta$ 为温度系数。

## 核心要点

1. **监督形式**：将 RL 优化转化为加权监督学习，高优势动作获得更大梯度权重
2. **Off-Policy 友好**：可从离线缓冲区直接采样，不依赖 on-policy rollout
3. **温度控制**：$\beta$ 越小，权重越集中于高优势样本；$\beta$ 越大，趋近于均匀 BC
4. **无策略比率裁剪**：相比 PPO 更简单，但需要高质量的价值函数估计

## 适用场景

- Off-policy VLA 微调（自回归 VLA，如 OpenVLA-OFT）
- 离线强化学习（offline RL）
- 数据效率要求高的场景

## 代表工作

- [[WCM]]: 使用 WCM 价值估计替代传统单帧 critic，提升 AWR 在 VLA-RL 中的效果
- [[SERL]]: 机器人学习中的 off-policy RL 框架

## 相关概念

- [[PPO]]：on-policy 对应算法，使用策略比率裁剪
- [[RECAP]]：流匹配 VLA 的 off-policy RL 对应方法
- [[行为克隆]]：AWR 的核心损失形式
- [[值函数]]：AWR 需要准确的价值估计来计算优势
