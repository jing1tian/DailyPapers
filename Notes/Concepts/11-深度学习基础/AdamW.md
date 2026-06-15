---
type: concept
aliases: [AdamW 优化器, Adam Weight Decay]
---

# AdamW

## 定义
AdamW 是 Adam 优化器的改进版本，将权重衰减（L2 正则化）从梯度更新中解耦，直接作用于参数本身。

## 数学形式
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

其中 $\lambda$ 为权重衰减系数，$\hat{m}_t, \hat{v}_t$ 分别为一阶和二阶矩估计。

## 核心要点
1. 标准 Adam 将权重衰减加入梯度，AdamW 直接对参数做衰减，行为更接近 SGD+momentum 的 L2 正则
2. 在 Transformer 类模型中广泛使用，是视觉、语言、机器人策略模型的默认优化器
3. 典型超参：lr=1e-4，weight_decay=0.01，betas=(0.9, 0.999)

## 代表工作
- Loshchilov & Hutter (2019)：原始 AdamW 论文

## 相关概念
- [[Adam]]
- [[SGD]]
- [[学习率调度]]
