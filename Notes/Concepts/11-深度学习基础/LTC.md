---
type: concept
aliases: [Liquid Time-Constant Network, 液态时间常数网络, LTC-RNN]
---

# LTC (Liquid Time-Constant Network)

## 定义
Liquid Time-Constant (LTC) 网络是一种连续时间递归神经网络，时间常数 $\tau$ 由输入动态决定（"液态"），使其能自适应地调整状态更新速率，适合建模非平稳时间序列。

## 数学形式
$$\tau(x) \frac{dh}{dt} = -h + f(h, x; \theta)$$

$$\tau(x) = \tau_{sys} \cdot \sigma(A x + b) + \tau_{min}$$

其中 $\tau(x)$ 是输入依赖的时间常数，$h$ 为隐状态。

## 核心要点
1. 与标准 LSTM/GRU 不同：时间常数由当前输入决定，而非固定
2. 适合机器人等"速率变化"场景（停止时慢更新，运动时快更新）
3. 可用作 episode-level progress 的 continuous-time embedding（TFP 论文用法）
4. 计算代价与 ODE Solver 精度正相关

## 代表工作
- [[TFP]]: 用 LTC 编码任务进度，条件化 visuomotor policy

## 相关概念
- [[POMDP]]
- [[State-Space Model]]
