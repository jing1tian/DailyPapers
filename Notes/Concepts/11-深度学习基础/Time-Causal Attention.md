---
type: concept
aliases: [时间因果注意力, Temporal Causal Attention, Causal Temporal Attention]
---

# Time-Causal Attention

## 定义

一种在序列 Transformer 中应用的注意力掩码策略：**同一时间步内的 tokens 允许双向交互，跨时间步则严格遵循因果顺序**（只能 attend 到当前及更早时间步，不能访问未来时间步信息）。

## 数学形式

$$
M_{ij} = \begin{cases} 0 & \text{if } \tau(j) \leq \tau(i) \\ -\infty & \text{if } \tau(j) > \tau(i) \end{cases}
$$

其中 $\tau(i)$ 为 token $i$ 所属的时间步；$M_{ij}=0$ 表示允许 attend，$M_{ij}=-\infty$ 表示屏蔽（future masking）。

## 核心要点

1. **时间步内双向**: 同一时间步的多个 tokens（如多视角的 patch tokens + register tokens）之间可以互相 attend，充分交换信息。
2. **跨时间步因果**: 时间步 $t$ 只能看到 $\leq t$ 的历史，保证推理时无需未来信息，支持在线/自回归推理。
3. **区别于纯因果掩码**: 标准 causal attention 对序列中每个 token 都因果；Time-Causal Attention 以时间步为粒度，步内双向、步间因果。

## 代表工作

- [[GWM-VLA]]: 在潜在世界模型中使用 Time-Causal Attention 处理多视角时序 token 序列

## 相关概念

- [[Block Causal Attention]]: 相关概念，以 block（块）为粒度的因果掩码
- [[JEPA]]: 自预测架构中常用的时序建模方式
