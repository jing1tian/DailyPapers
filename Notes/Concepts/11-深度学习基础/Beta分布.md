---
type: concept
aliases: [Beta Distribution, Beta分布, 贝塔分布]
---

# Beta 分布

## 定义
Beta 分布是定义在区间 $(0,1)$ 上的连续概率分布，由形状参数 $\alpha > 0$ 和 $\beta > 0$ 控制，常用于建模概率值或比例。

## 数学形式

$$
p(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in (0,1)
$$

其中 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ 为 Beta 函数。

**众数（Mode）**（当 $\alpha, \beta > 1$ 时）：

$$
m = \frac{\alpha - 1}{\alpha + \beta - 2}
$$

**用 Mode $m$ 和集中度 $c$ 反参数化（$c > 2$）**：

$$
\alpha = m(c-2) + 1, \quad \beta = (1-m)(c-2) + 1
$$

## 核心要点
1. **支撑域为 $(0,1)$**: 天然适合建模概率、比例、保留率等有界量
2. **灵活形状**: $\alpha=\beta=1$ 时退化为均匀分布；$\alpha,\beta > 1$ 时为单峰；$\alpha,\beta < 1$ 时为 U 形
3. **众数参数化**: 在神经网络输出中，用众数 $m$（sigmoid 输出）和集中度 $c$ 参数化比直接输出 $\alpha,\beta$ 更稳定

## 代表工作
- [[SANTS]]: 用 Beta 分布对视频去噪的噪声保留比 $r_k \in (0,1)$ 建模，实现自适应噪声跳进

## 相关概念
- [[流匹配]]: 常与 Beta 分布结合，用于连续时间噪声调度
- [[扩散策略]]: 噪声调度是扩散策略推理的核心环节
