---
type: concept
aliases: [噪声权重, 时间步权重, Timestep Weighting, Loss Weighting]
---

# Noise Weighting

## 定义
噪声权重是在 [[Flow Matching]] 或扩散模型训练中，对不同噪声水平（时间步）的损失给予不同权重的技术，通常对中间噪声水平给予更高权重以提升训练稳定性。

## 数学形式

PrismWAM 中的噪声权重函数：

$$
w(\sigma) = \frac{y(\sigma) - y_{\min}}{\bar{y} - y_{\min} + 10^{-10}}, \quad y(\sigma) = \exp\left[-2\left(\sigma - \frac{1}{2}\right)^2\right]
$$

其中 $y_{\min}$ 和 $\bar{y}$ 在 1000 个均匀网格点上计算，以 $\sigma=0.5$ 为峰值的高斯权重。

## 核心要点

1. **中间时间步权重最高**: 过高（纯噪声）和过低（干净）噪声水平的梯度信号质量较差
2. **归一化**: 将权重归一化到 $[0, 1]$ 区间，避免量纲问题
3. **多模态适用**: 各模态（RGB、深度、动作）独立采样噪声水平并分别加权
4. **与 EDM、Stable Diffusion 3 类似**: 不同框架有不同权重函数设计

## 代表工作
- [[PrismWAM]]: 对三个模态分别应用噪声权重的联合 Flow Matching 训练

## 相关概念
- [[Flow Matching]]
- [[Joint Training Objective]]
