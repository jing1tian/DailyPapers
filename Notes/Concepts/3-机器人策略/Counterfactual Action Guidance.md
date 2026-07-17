---
type: concept
aliases: [CAG, 反事实动作引导]
---

# Counterfactual Action Guidance (CAG)

## 定义

一种推理时的双分支策略增强方案：将原始 VLA 的有语言条件输出与无语言条件输出做加权外推，以类比 Classifier-Free Guidance 的方式放大语言信号，无需修改模型架构。

## 数学形式

$$
\pi_{\mathrm{CAG}}(a \mid o, l) = \pi_{\mathrm{uncond}}(a \mid o, \emptyset) + \omega \cdot \bigl(\pi_{\mathrm{cond}}(a \mid o, l) - \pi_{\mathrm{uncond}}(a \mid o, \emptyset)\bigr)
$$

等价 Bayesian 形式：

$$
P_{\mathrm{CAG}}(a \mid o, l) \propto P(a \mid o) \cdot P(l \mid a, o)^{\omega}
$$

## 核心要点

1. 从 [[贝叶斯推断]] 分析 VLA 视觉捷径：训练偏置导致 $p(a|o,l) \approx p(a|o)$
2. 两种实现：Training-Free（置空语言）和 Vision-Action（独立训练无条件分支）
3. [[引导尺度]] ω > 1 放大语言信号，过大会导致过度引导
4. 与架构无关，已在 autoregressive 和扩散 VLA 上验证

## 代表工作

- [[CAG]]: 原始提出论文，UNC Chapel Hill，2026

## 相关概念

- [[Classifier-Free Guidance (CFG)]]
- [[VLA（视觉-语言-动作模型）]]
- [[引导尺度]]
- [[贝叶斯推断]]
