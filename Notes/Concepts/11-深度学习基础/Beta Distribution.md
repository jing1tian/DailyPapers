---
type: concept
aliases: [Beta分布, Beta Distribution, 贝塔分布]
---

# Beta Distribution

## 定义

Beta 分布是定义在 $[0,1]$ 区间上的连续概率分布，由两个形状参数 $\alpha, \beta > 0$ 控制，广泛用于概率建模和噪声采样。

## 数学形式

$$
f(x;\alpha,\beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in [0,1]
$$

其中 $B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ 为 Beta 函数（归一化常数）。

## 核心要点

1. $\alpha > \beta$ 时分布偏向 1（高值），$\alpha < \beta$ 时偏向 0（低值）
2. $\alpha = \beta = 1$ 时退化为均匀分布
3. **Beta(7,1)**: 集中在接近 1 处，用于高噪声采样
4. **Beta(1,7)**: 集中在接近 0 处，用于低噪声采样

## 代表工作

- [[LoopWAM]]: 使用 Beta(7,1) 和 Beta(1,7) 实现源相关噪声采样，区分专家段和 rollout 段的训练信号

## 相关概念

- [[Flow Matching]]
- [[Beta分布]]
