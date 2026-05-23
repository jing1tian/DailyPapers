---
type: concept
aliases: [Block Causal Attention Mask, 块因果注意力, 块因果掩码]
---

# Block Causal Attention

## 定义

一种 [[Transformer]] 注意力掩码策略：同一时间步（帧）内的所有 token 互相可见（形成注意力"块"），跨时间步则只能看到历史帧，从而实现帧内并行预测与跨帧因果约束的结合。

## 数学形式

对于帧序列 $s_1, s_2, \ldots, s_T$，每帧含 $L$ 个 token，注意力掩码定义为：

$$
M_{ij} = \begin{cases} 1 & \text{if } t(i) = t(j) \text{ or } t(i) > t(j) \\ 0 & \text{if } t(i) < t(j) \end{cases}
$$

即同帧（$t(i) = t(j)$）token 互相 attend，历史帧（$t(i) > t(j)$）也可 attend，但不能 attend 未来帧。

## 核心要点

1. **帧内并行**: 与标准逐 token causal mask 不同，同帧所有 token 同时处理，无需顺序生成
2. **跨帧因果**: 保留跨帧的时间方向因果性，不会利用未来信息
3. **计算效率**: 允许帧级别的并行预测，显著加速 rollout 生成
4. **空间上下文**: 帧内 token 互相可见，可利用空间上下文信息预测每个位置

## 代表工作

- [[ITC]]: 使用 Block Causal Attention 结合 3D RoPE，实现帧内并行 token 预测

## 相关概念

- [[Transformer]]: 基础架构
- [[3D RoPE]]: 常与 Block Causal Attention 配合，提供时空位置感知
- [[World Model]]: Block Causal Attention 的主要应用场景
