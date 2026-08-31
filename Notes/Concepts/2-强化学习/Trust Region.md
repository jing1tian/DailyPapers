---
type: concept
aliases: [信任域, 信任区域, Trust Region Constraint, 置信域]
---

# Trust Region

## 定义

Trust Region（信任域）：约束优化中限制每次参数/输出更新幅度的球形区域，保证一阶近似在其内部足够精确，从而使保守的策略改进步骤有理论保证。

## 数学形式

在 DiffusionOPSD 的目标构建中，trust-region 约束为：

$$
\|y^{(m)}_\pm - y_0\|_2 \leq \rho\|y_0\|_2
$$

投影算子将超出边界的目标投影回球内：

$$
\Pi_r(d) = \begin{cases} d, & \|d\|_2 \leq r \\ r\,d/\|d\|_2, & \|d\|_2 > r \end{cases}
$$

经典 TRPO 中的 KL 散度约束：

$$
\max_\theta\,\mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}A(s,a)\right], \quad \text{s.t.}\; D_{KL}(\pi_{old}\|\pi_\theta) \leq \delta
$$

## 核心要点

1. **保守改进**：只在局部一阶近似可靠的范围内移动，避免策略崩溃
2. **DiffusionOPSD 中的应用**：以 $\rho\|y_0\|_2$ 为半径限制 clean-output 目标的移动，保证 reward 局部近似有效
3. **几何意义**：半径 $\rho$ 控制目标与 anchor 的最大相对位移（$\rho=0.10$ 为默认值）
4. **与 RL 的联系**：信任域思想源自 TRPO/PPO，DiffusionOPSD 将其应用于 clean-output 空间

## 代表工作

- TRPO（Schulman et al., 2015）：经典 trust-region 策略优化
- PPO：近似 trust-region 约束的 clipping 实现
- [[DiffusionOPSD]] (2608.24646)：在 clean-output 空间的 trust-region 目标构建

## 相关概念

- [[RLHF]]
- [[Behavior Policy]]
- [[On-Policy Distillation]]
- [[DiffusionOPSD]]
