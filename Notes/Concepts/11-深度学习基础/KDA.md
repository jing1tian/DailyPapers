---
type: concept
aliases: [Kernel Delta Attention, Gated Delta Rule]
---

# KDA

## 定义
Kernel Delta Attention：线性注意力的一种变体，通过 Delta 规则在固定大小的循环状态上进行精确关联写入，避免 softmax attention 的 $O(L^2)$ 复杂度。

## 数学形式
状态更新（Delta 规则）：
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{v}_t \otimes \mathbf{k}_t - (\mathbf{S}_{t-1}\mathbf{k}_t) \otimes \mathbf{k}_t$$

Gated DeltaNet-2 解耦版（erase/write 独立门控）：
$$\mathbf{S}_t = \mathbf{g}_t^e \odot \mathbf{S}_{t-1} + \mathbf{g}_t^w \odot (\mathbf{v}_t \otimes \mathbf{k}_t)$$

## 核心要点
1. 线性时间复杂度 $O(L)$，解码时常数内存
2. Delta 规则精确编辑状态，不影响无关关联
3. 可推导等价的 chunkwise 计算形式，适合并行训练
4. Gated DeltaNet-2 进一步解耦 erase gate 和 write gate，解决 DeltaNet-1 写新内容时破坏旧关联的问题

## 代表工作
- [[Gated-DeltaNet-2]]（Hatamizadeh et al., 2026）: erase/write 解耦改进
- DeltaNet（Schlag et al., 2021）: 原始 Delta 规则线性注意力

## 相关概念
- [[Mamba]]
- [[SSM]]
- [[Transformer]]
