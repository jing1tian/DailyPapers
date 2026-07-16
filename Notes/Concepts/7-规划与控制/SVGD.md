---
type: concept
aliases: [Stein Variational Gradient Descent, 斯坦变分梯度下降]
---

# SVGD

## 定义
SVGD（Stein Variational Gradient Descent）是用于近似贝叶斯推断的粒子优化方法，通过在 RKHS 中最小化 Stein discrepancy 将一组粒子迭代驱动至目标分布，同时保持粒子多样性。

## 数学形式
$$
\phi^*(x) = \mathbb{E}_{x' \sim q}\left[k(x', x) \nabla_{x'} \log p(x') + \nabla_{x'} k(x', x)\right]
$$

梯度方向 $\phi^*$ 驱动粒子降低 KL 散度；核函数 $k$ 提供排斥力保证多样性。

## 核心要点
1. **无需 MCMC 链**：批量粒子并行更新，速度比 MCMC 快
2. **多样性保持**：RBF 核的排斥项阻止粒子坍缩到单一模式
3. **确定性**：同等条件下结果可重复，而非随机游走
4. **局限**：高维时核带宽选择困难，收敛性理论保证有限

## 代表工作
- [[SteinSQP]]: 将 SVGD 与 SQP 结合，约束条件下采样多样化可行运动轨迹

## 相关概念
- [[MPPI]]
- [[CEM]]
