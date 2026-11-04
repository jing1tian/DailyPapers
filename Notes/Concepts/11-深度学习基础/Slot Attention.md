---
type: concept
aliases: [Slot Attention, 槽位注意力, Object-Centric Learning]
---

# Slot Attention

## 定义
一种迭代注意力机制，将视觉特征映射为固定数量的"槽位（slot）"，每个槽位竞争性地绑定到场景中一个对象或区域，实现无监督的对象中心表示学习。

## 数学形式

$$
\text{Attn}(k, q) = \text{softmax}\!\left(\frac{k \cdot q^T}{\sqrt{d}}, \text{dim=slots}\right)
$$

$$
s \leftarrow \text{GRU}\!\left(s,\; \text{WeightedMean}(\text{Attn}, v)\right)
$$

通过跨槽位的归一化（softmax over slots），每个输入特征被竞争性分配给各槽位，迭代 $T$ 次收敛。

## 核心要点
1. **竞争性绑定**: softmax 在槽位维度归一化，迫使各槽位专注于不同区域
2. **排列不变**: 槽位无固定顺序，满足对象中心表示的对称性要求
3. **迭代细化**: 通过多轮 attention + GRU 更新，逐步提升绑定质量
4. **下游应用**: 机器人操作场景重建（GHOST/SCOUT）、视频理解、3D 场景分解

## 代表工作
- [[SCOUT]]: 以 Slot Transformer 细化对象中心槽位，重建遮挡场景

## 相关概念
- [[GHOST]]
- [[Transformer]]
- [[Self-Attention]]
- [[SlotContrast]]
- [[SlotLSTM]]
