---
type: concept
aliases: [Mixture of Block Attention, 分块混合注意力]
---

# MoBA

## 定义
一种高效长序列注意力机制，将输入序列分成若干块（block），每个 query 只选择性地关注部分块（mixture），而非计算全局 attention，从而降低长序列的计算复杂度。

## 数学形式
$$\text{MoBA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \sum_{b \in \text{selected blocks}} \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}_b^\top}{\sqrt{d}}\right)\mathbf{V}_b$$

## 核心要点
1. 将 KV 序列分块，用 routing 机制动态选择每个 query 关注哪些块
2. 计算复杂度从 $O(n^2)$ 降低到 $O(n \cdot k \cdot B)$，$k$ 为选中块数，$B$ 为块大小
3. 适用于世界模型、视频生成等需要处理超长 context 的场景

## 代表工作
- [[World-Infinity]]: 使用 MoBA 支持无限时长视频序列的 attention 计算

## 相关概念
- [[DiT]]
- [[MoE]]
- [[AdaLN]]
