---
type: concept
aliases: [Batch-Constrained Q-learning, 批约束Q学习]
---

# BCQ

## 定义
Batch-Constrained Q-learning：离线强化学习算法，通过将策略的动作空间约束在行为数据集支撑集（batch support）内，解决分布偏移（distributional shift）问题。

## 数学形式
$$\pi(s) = \arg\max_{a: G_\omega(a|s) \geq \tau} Q_\theta(s, a)$$

其中 $G_\omega$ 是条件 VAE 生成器，只生成数据集内存在的动作；$\tau$ 是约束阈值。

## 核心要点
1. 核心问题：离线 RL 中，策略访问数据集外的 (s,a) 对时 Q 值会被高估（extrapolation error）
2. 解法：用 conditional VAE 学习行为策略，只在生成的候选动作中选 Q 值最大的
3. 引入 perturbation model 在约束范围内小幅调整动作，平衡约束与性能
4. 是最早系统解决 offline RL 分布偏移的方法之一

## 代表工作
- [[BCQ]]: Fujimoto et al., "Off-Policy Deep RL without Exploration", ICML 2019

## 相关概念
- [[IQL]], [[CQL]], [[TD3+BC]]: 后续 offline RL 算法
- [[CMDP]]: 安全约束下的 offline RL 扩展
