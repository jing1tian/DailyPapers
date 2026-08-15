---
type: concept
aliases: [Jensen-Shannon Divergence, JSD, JS散度, Jensen-Shannon距离]
---

# Jensen-Shannon 散度 (Jensen-Shannon Divergence)

## 定义

KL 散度的对称化版本：对两个概率分布 $P$ 和 $Q$，以其混合分布 $M = \tfrac{1}{2}(P+Q)$ 为中间量，计算各自到 $M$ 的 KL 散度均值，得到有界、对称的统计距离。

## 数学形式

$$
\mathrm{JSD}(P \| Q) = \tfrac{1}{2}\,D_{\mathrm{KL}}(P \| M) + \tfrac{1}{2}\,D_{\mathrm{KL}}(Q \| M), \quad M = \tfrac{1}{2}(P + Q)
$$

JSD 取值范围 $[0, \ln 2]$（nat 单位）或 $[0, 1]$（bit 单位），其平方根构成度量空间。

## 核心要点

1. **对称性**: $\mathrm{JSD}(P\|Q) = \mathrm{JSD}(Q\|P)$，克服 KL 散度非对称的缺陷。
2. **有界性**: 不像 KL 散度可趋向无穷，JSD 有上界，梯度更稳定。
3. **支撑集兼容**: 即使 $P$、$Q$ 支撑集不完全重叠，JSD 仍有限（KL 散度此时为无穷）。
4. **知识蒸馏应用**: 在教师-学生蒸馏中常用来替代单向 KL，使师生双向对齐。

## 代表工作

- [[FIRE-VLA]]: 在失败感知自蒸馏中以 token 级 JSD 度量教师与学生输出分布差异，并设 0.05 上限截断防止梯度爆炸

## 相关概念

- [[KL 散度]]
- [[知识蒸馏]]
- [[自蒸馏]]
