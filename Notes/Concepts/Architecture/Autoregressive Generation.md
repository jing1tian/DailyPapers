---
type: concept
aliases: [自回归生成, 自回归预测, Autoregressive Prediction]
---

# Autoregressive Generation

## 定义

将序列生成分解为逐步条件预测，每一步以历史生成结果为条件预测下一步，实现长时域序列的递推生成。

## 数学形式

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})
$$

在世界模型中的应用：

$$
p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k}, \mathcal{H}_{t-1})
$$

## 核心要点

1. **误差累积**: 每步预测误差会在后续步骤中累积，是长程生成退化的主要原因
2. **历史条件**: 引入历史帧（$\mathcal{H}_{t-1}$）可缓解误差累积，提供上下文参考
3. **Self-forcing 缓解**: 训练时随机以自身预测帧替换真实帧，暴露模型于自身误差（见 [[Self-forcing]]）

## 代表工作

- [[A2World]]: 采用自回归生成 + 姿态引导历史采样实现稳定长程世界模型仿真
- Transformer (GPT): 语言领域的经典自回归生成

## 相关概念

- [[Self-forcing]]
- [[Action-Conditioned World Model]]
- [[Pose-guided History Sampling]]
