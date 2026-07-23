---
type: concept
aliases: [Convolutional LSTM, ConvLSTM2D]
---

# ConvLSTM

## 定义
将 LSTM 中的矩阵乘法替换为卷积操作的时序模型，保留空间局部性，适合时空序列预测。

## 数学形式
$$\begin{aligned}
i_t &= \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + b_f) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tanh(W_{xc} * X_t + W_{hc} * H_{t-1} + b_c) \\
H_t &= o_t \odot \tanh(C_t)
\end{aligned}$$

其中 $*$ 表示卷积，$\odot$ 为逐元素乘积。

## 核心要点
1. Hidden state $H_t$ 保持空间维度，适合格点型 state 的时序预测
2. 比标准 LSTM 参数量更少（权重共享）
3. 在 RL 中用作 recurrent 状态表示，比 MLP+LSTM 更保留空间结构
4. 与 [[SlotLSTM]] 的区别：ConvLSTM 无明确的 slot binding，缺乏 relational 结构

## 代表工作
- Shi et al. 2015: 原始 ConvLSTM 用于降水预报
- [[DRC]]（Deep Repeated ConvLSTM）: 在 Sokoban 等任务上的 RL 应用

## 相关概念
- [[SlotLSTM]]（relational hidden state 变体）
- [[AttnLSTM]]（注意力增强的 LSTM）
- [[DRC]]（深度重复 ConvLSTM）
