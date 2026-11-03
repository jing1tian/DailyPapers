---
type: concept
aliases: [Teacher Forcing, 教师强制, 导师强制]
---

# Teacher Forcing

## 定义

序列生成模型的一种训练技术：训练时用真实目标序列（ground-truth tokens）作为解码器的输入，而非模型自身上一步的预测输出；推理时则自回归地使用模型自身预测。通过减少训练时的误差累积，加速收敛并稳定训练。

## 数学形式

设序列 $y = (y_1, \ldots, y_T)$，模型参数 $\theta$：

- **Teacher Forcing 训练**: $p_\theta(y_t \mid y_{<t}^*)$，其中 $y_{<t}^*$ 为真实前缀
- **自回归推理**: $p_\theta(y_t \mid \hat{y}_{<t})$，其中 $\hat{y}_{<t}$ 为模型预测

## 核心要点

1. **暴露偏差问题（Exposure Bias）**: Teacher Forcing 导致训练/推理分布不匹配——训练时见真实 token，推理时见自身预测，可能导致误差累积
2. **加速收敛**: 避免早期训练时错误预测导致梯度信号极差的问题
3. **Scheduled Sampling**: 缓解暴露偏差的改进——训练时以概率 $p$ 使用真实 token，$1-p$ 使用模型预测
4. **扩展应用**: 在扩散/流匹配模型中衍生出 [[AR-Forcing]]、[[动作强制]] 等变体

## 代表工作

- 广泛用于 RNN/LSTM/Transformer 序列生成任务（语言模型、机器翻译、图像 captioning）

## 相关概念

- [[AR-Forcing]]
- [[动作强制]]
- [[Autoregressive Policy]]
