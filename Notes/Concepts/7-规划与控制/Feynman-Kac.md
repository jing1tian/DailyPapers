---
type: concept
aliases: [Feynman-Kac公式, 路径积分控制]
---

# Feynman-Kac

## 定义
物理/数学中的路径积分公式，将偏微分方程的解表示为随机过程的期望值。在控制领域，被用于路径积分控制（Path Integral Control），将最优控制问题转化为对随机轨迹分布的采样问题。

## 数学形式
Feynman-Kac 公式：
$$u(x, t) = \mathbb{E}\left[\int_t^T f(X_s, s)ds + g(X_T) \mid X_t = x\right]$$
在随机控制中，最优策略等价于对加权轨迹分布的期望：
$$\pi^*(a|s) \propto \exp\left(\frac{1}{\lambda} \sum_t r(s_t, a_t)\right)$$

## 核心要点
1. **控制即推断**：将最优控制问题转化为概率推断
2. **随机采样**：用蒙特卡洛方法估计最优控制信号
3. **不确定性建模**：天然支持对人类意图等不确定量建模

## 代表工作
- [[CollaborativeVLA]] (2606.12475): 用 Feynman-Kac 路径积分建模人类意图不确定性

## 相关概念
- [[MPC]]
- [[CEM]]
- [[PPO]]
