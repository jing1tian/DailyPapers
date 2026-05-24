---
type: concept
aliases: [EMA, Exponential Moving Average, Momentum Encoder, Target Network, 指数移动平均]
---

# Exponential Moving Average (EMA)

## 定义
对参数的滑动指数平均：$\theta_{ema} \leftarrow m \theta_{ema} + (1-m)\theta$。常用于 SSL（target encoder）、扩散模型（参数 EMA 推理）、RL（target network）。

## 数学形式
$$
\theta^{(t+1)}_{ema} = m\,\theta^{(t)}_{ema} + (1-m)\,\theta^{(t)},\quad m \in [0.99, 0.9999]
$$

## 核心要点
1. SSL 中作为 target 提供稳定预测目标，避免崩塌。
2. 是 BYOL / DINO / I-JEPA 的"非对称"关键。
3. [[LeWM]] 显式不用 EMA：靠 [[SIGReg]] 已经避免崩塌。

## 代表工作
- BYOL, MoCo, DINO, I-JEPA
- [[LeWM]]：表明 EMA 不是必需的
- [[SOMA]]：使用自适应 EMA 实现空间记忆的动态精炼，融合系数由语义相似度自动调节

## 相关概念
- [[Stop Gradient]]
- [[Representation Collapse]]
