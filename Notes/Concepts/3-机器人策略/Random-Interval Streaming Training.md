---
type: concept
aliases: [随机间隔流式训练, RIST]
---

# Random-Interval Streaming Training

## 定义

一种 VLA 训练策略，在训练时对帧间时间间隔引入均匀分布随机扰动，使模型同时适应同步训练环境和异步真实机器人部署环境，解决训练-部署时序分布偏移问题。

## 数学形式

$$
\delta = \bar{\delta} + \epsilon, \quad \epsilon \sim \mathcal{U}(-\Delta, +\Delta)
$$

**符号说明**：
- $\delta$：实际帧间时间间隔
- $\bar{\delta}$：基础间隔（中心值）
- $\epsilon$：均匀分布随机扰动，范围 $[-\Delta, +\Delta]$
- 实际训练范围：$\delta \sim \mathcal{U}[3, 7]$

配合**时序掩码**（temporal masking）模拟增量观测模式。

## 核心要点

1. 真实机器人系统中相机帧率与控制频率异步，固定间隔训练会导致部署时的分布外失败
2. 随机间隔训练将不同时序间距都纳入训练分布，提高模型对帧率变化的鲁棒性
3. 在 LIBERO 上相比固定间隔训练额外带来 +1.1%（T=3）和 +1.3%（T=5）的成功率提升
4. 训练好的模型可以跨 T 值泛化（T=5 训练，T=3 测试性能损失极小）

## 代表工作

- [[StreamPI]]: 提出该训练策略，配合 [[Instruction-Anchored Temporal Modeling]] 使用

## 相关概念

- [[Instruction-Anchored Temporal Modeling]]
- [[KV Cache]]
- [[π0.5]]
