---
type: concept
aliases: [孔径参数, α]
---

# Aperture Parameter（孔径参数）

## 定义

Conceptor 的正则化超参数 $\alpha$，控制 Conceptor 对激活子空间的覆盖范围：$\alpha$ 越大，覆盖越宽；$\alpha$ 越小，覆盖越窄（更严格的低秩近似）。

## 数学形式

$$
\mu_i = \frac{\lambda_i}{\lambda_i + \alpha^{-2}}
$$

- $\alpha \to \infty$：$\mu_i \to 1$（Conceptor 趋近单位矩阵，覆盖全部方向）
- $\alpha \to 0$：$\mu_i \to 0$（Conceptor 趋近零矩阵，不保留任何方向）

## 核心要点

1. 孔径参数是 Conceptor 最关键的超参数，直接影响子空间覆盖范围
2. COAST 的三阶段超参数选择中，第二阶段通过成功-失败子空间重叠最大化选择最优 $\alpha$
3. 对比不同 aperture 的 Conceptor 可以做 Conceptor 代数运算

## 代表工作

- [[COAST]]: 通过高效三阶段启发式为 $\alpha$ 选取最优值

## 相关概念

- [[Conceptor]]
- [[Subspace Similarity]]
