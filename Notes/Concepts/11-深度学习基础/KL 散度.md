---
type: concept
aliases: [KL Divergence, Kullback-Leibler Divergence, 相对熵]
---

# KL 散度 (KL Divergence)

## 定义

KL 散度（Kullback-Leibler Divergence）衡量两个概率分布之间的差异程度，常用于策略优化中约束新策略与旧策略不要偏离太远（信赖域约束），或用于变分推断中拟合分布与目标分布的逼近误差。

## 数学形式

$$
D_{KL}(p \,\|\, q) = \mathbb{E}_{x\sim p}\left[\log \frac{p(x)}{q(x)}\right]
$$

## 核心要点

1. **非对称性**：$D_{KL}(p\|q) \neq D_{KL}(q\|p)$，方向选择会影响优化行为（模式覆盖 vs. 模式寻找）
2. **非负性**：$D_{KL}(p\|q) \geq 0$，当且仅当 $p=q$ 时取零
3. **策略优化中的应用**：作为正则项约束策略更新幅度，如 [[近端策略优化|PPO]]、MPO、[[VGPD]] 等方法中限制新策略 $\pi$ 与旧策略 $\pi_{old}$ 的偏离程度
4. **能量加权解**：在带 KL 约束的策略改进目标中，解析最优解通常呈现 $\pi^* \propto \pi_{old}\exp(Q/\tau)$ 的能量加权形式

## 代表工作

- [[VGPD]]: 用 KL 散度约束策略改进目标，推导出能量加权自蒸馏更新规则
- [[近端策略优化]]: 用 KL 散度（或其近似）约束新旧策略差异，保证训练稳定性

## 相关概念

- [[近端策略优化]]
- [[Actor-Critic]]
- [[强化学习]]
