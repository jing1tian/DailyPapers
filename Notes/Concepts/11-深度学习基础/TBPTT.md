---
type: concept
aliases: [Truncated BPTT, 截断反向传播]
---

# TBPTT

## 定义
Truncated Backpropagation Through Time（截断的时间反向传播），在序列长度上限制梯度反向传播的窗口以避免梯度消失/爆炸，同时保留跨步骤的隐状态前向传递。

## 数学形式
给定序列 $x_{1:T}$，TBPTT 将其分为长度为 $K$ 的子序列，每段独立计算梯度：
$$\nabla_\theta \mathcal{L} = \sum_{t} \nabla_\theta \mathcal{L}_t, \quad \text{grad window} = K$$

## 核心要点
1. **前向不截断**：隐状态 $h_t$ 从 $t=0$ 连续传递，保留完整上下文
2. **反向截断**：梯度只在最近 $K$ 个时间步内传播
3. **K 的选择**：K 过小导致长程依赖学习不足，K 过大接近完整 BPTT 的计算代价

## 代表工作
- [[μVLA]]: 用 TBPTT 训练 VLA 的循环记忆 token，K 是关键超参数
- [[DreamerV3]]: 在潜在空间序列上使用类似截断方法

## 相关概念
- [[RSSM]]
- [[JEPA]]
