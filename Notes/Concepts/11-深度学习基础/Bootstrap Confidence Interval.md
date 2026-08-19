---
type: concept
aliases: [自举置信区间, 场景自举, bootstrap CI, 95% CI, bootstrap confidence interval]
---

# Bootstrap Confidence Interval（自举置信区间）

## 定义

**自举置信区间（Bootstrap CI）**是一种非参数统计方法，通过对观测数据进行有放回重采样（bootstrapping）来估计统计量的抽样分布，从而构造置信区间，无需假设数据的分布形式。

## 数学形式

给定 $n$ 个观测样本 $\{x_1, \ldots, x_n\}$ 和统计量 $\hat{\theta}$：

$$
\text{CI}_{95\%} = [\hat{\theta}^{(2.5\%)}_{\text{boot}},\; \hat{\theta}^{(97.5\%)}_{\text{boot}}]
$$

其中 $\hat{\theta}^{(p)}_{\text{boot}}$ 为 $B$ 次自举样本计算的统计量分布的第 $p$ 百分位数。

## 核心要点

1. **无分布假设**: 适用于小样本（如 40 个场景）和非正态数据
2. **"已解决" vs "未解决"**: 若 95% CI 不包含零，效果"已解决"（统计显著）；CI 包含零则"未解决"（不确定）
3. **场景级独立单元**: 在视频生成评估中，以场景为独立单元进行自举，避免伪重复
4. **保守估计**: 自举 CI 通常比参数方法（t 检验）保守，对小样本实验尤为重要

## 代表工作

- [[SCOPE]]: 使用场景自举 95% CI 报告所有实验结果，明确区分统计上已解决和未解决的效果

## 相关概念

- [[Score Isolation]]
