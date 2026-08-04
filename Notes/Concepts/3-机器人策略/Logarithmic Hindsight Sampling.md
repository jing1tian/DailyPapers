---
type: concept
aliases: [对数回顾采样, 对数历史采样, Logarithmic Sampling]
---

# Logarithmic Hindsight Sampling（对数回顾采样）

## 定义

一种用于机器人操作策略的时序帧采样策略：以对数间隔从历史帧序列中选取关键帧，近期帧密集采样（捕捉精细动作变化），远期帧稀疏采样（提供任务级语义上下文），通过斐波那契递推约束消除离散化冗余。

## 数学形式

**基础对数采样**:

$$
k_i = \lfloor q_{min} \cdot r^i \rfloor
$$

**斐波那契稀疏约束**（消除 index collision）:

$$
k_i \geq k_{i-1} + k_{i-2}, \quad \forall i > 2
$$

其中 $q_{min}$ 为最小采样间隔，$r > 1$ 为增长率，$k_i$ 为相对当前时刻的历史偏移步数。

随采样深度增加，相邻索引比趋近黄金分割比 $\phi \approx 1.618$，自然维持对数分布。

## 核心要点

1. **近密远疏**: 符合机器人操作中时序信息密度的非均匀特性
2. **双重设计**: 对数采样保证信息覆盖，斐波那契约束消除离散化冗余
3. **推理友好**: 斐波那契约束副作用——为 [[Fibonacci Recurrent Inference|Fibonacci 递归推理]] 中的 KV Cache 复用创造数学前提

## 代表工作

- [[FibVLA]]: 提出并结合 CTE 模块与 Fibonacci 递归推理使用

## 相关概念

- [[Fibonacci Recurrent Inference]]: 利用斐波那契约束实现 KV Cache 复用
- [[Channel-wise Temporal Encoding]]: 对采样帧进行时序特征编码
- [[Action Chunking]]: 动作块长度 $L$ 与斐波那契采样绑定
