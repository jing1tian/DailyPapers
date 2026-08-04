---
type: concept
aliases: [斐波那契递归推理, Fibonacci推理, Fibonacci Recurrent]
---

# Fibonacci Recurrent Inference（斐波那契递归推理）

## 定义

利用斐波那契数列的递推性质，使 VLA 在相邻推理步骤之间精确复用历史帧的 [[KV Cache]] 特征 token，避免对已编码历史帧的重复计算，从而在保持时序感知能力的同时大幅降低推理延迟。

## 数学形式

当采样索引满足 $k_i = k_{i-1} + k_{i-2}$，且 [[Action Chunking|动作块]] 长度 $L = k_{i-2}$ 时，时间步从 $t$ 推进到 $t' = t + L$ 后有：

$$
t' - k_i = (t + k_{i-2}) - k_i = t - k_{i-1}
$$

即：$t$ 时刻回溯 $k_{i-1}$ 步的物理帧 = $t'$ 时刻回溯 $k_i$ 步的物理帧（同一帧）。

所有高频（近期）更新仅发生在新时间窗口 $[t, t+L]$ 内，历史部分的 KV Cache token 完全对齐，可直接复用。

## 核心要点

1. **精确对齐**: 斐波那契约束保证历史 token 在下一步精确命中采样点（无近似）
2. **选择性更新**: 仅重新编码 $[t, t+L]$ 窗口内的新帧，历史帧 KV Cache 不变
3. **效率收益**: FibVLA 实测推理延迟 177ms，比 HiF-VLA 低 27.16%，比 TraceVLA 低 9.69%
4. **黄金比例稳定性**: 随采样深度增加，相邻间隔比趋近黄金分割比 $\phi$，保持对数分布不变形

## 代表工作

- [[FibVLA]]: 提出斐波那契递归推理，与 [[Logarithmic Hindsight Sampling]] 绑定设计

## 相关概念

- [[Logarithmic Hindsight Sampling]]: 其斐波那契稀疏约束为本方法提供数学前提
- [[KV Cache]]: 递归推理复用的底层机制
- [[Action Chunking]]: 动作块长度 $L$ 必须与 $k_{i-2}$ 绑定才能保证对齐
- [[Channel-wise Temporal Encoding]]: 协同处理历史帧的时序特征编码模块
