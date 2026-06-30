---
type: concept
aliases: [门控循环单元, Gated Recurrent Unit, GRU]
---

# GRU（门控循环单元）

## 定义

GRU 是一种循环神经网络单元，通过重置门和更新门控制信息流动，能够在序列建模中高效捕获长程依赖，同时避免梯度消失问题。

## 数学形式

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])
$$

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

其中 $\mathbf{r}_t$ 为重置门，$\mathbf{z}_t$ 为更新门，$\tilde{\mathbf{h}}_t$ 为候选隐状态。

## 核心要点

1. **重置门** $\mathbf{r}_t$：控制上一时刻隐状态对候选隐状态的影响程度，为 0 时完全遗忘历史
2. **更新门** $\mathbf{z}_t$：控制新旧隐状态的混合比例，相当于 LSTM 的遗忘门和输入门的合并
3. **比 LSTM 更轻量**：参数量少约 25%，适合嵌入在大型模型中作辅助模块

## 代表工作

- [[S2-VLA]]: 使用 GRU 参数化信念状态更新函数 $f_\phi$，将历史动作序列和本体感知压缩为任务进度表示

## 相关概念

- [[信念状态]]
- [[LSTM]]
- [[Transformer]]
