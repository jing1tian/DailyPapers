---
type: concept
aliases: [联合注意力, Cross-Expert Attention]
---

# Joint Attention

## 定义

在 [[Mixture-of-Transformers]] 多专家架构中，每个专家的注意力头同时关注所有专家的键值对（包括自身），实现专家间的双向信息交换。

## 数学形式

设专家集合 $m \in \{V, A\}$（视觉、动作）：

$$
H_m = \operatorname{Attn}\left(Q_m,\; [K_V;\, K_A],\; [V_V;\, V_A]\right)
$$

每个专家用自身的查询矩阵 $Q_m$，但键值矩阵拼接自所有专家流。

## 核心要点

1. **双向信息流**：视觉专家可读取动作 token，动作专家可读取视觉 token
2. **参数独立**：每个专家保持各自的 $Q, K, V$ 投影矩阵，共享计算但不共享参数
3. **与 Cross-Attention 的区别**：Cross-Attention 是单向（一方读取另一方），Joint Attention 是双向全融合

## 代表工作

- [[NAVA-WAM]]: 预训练阶段 Video-DiT 和 Action-DiT 通过 Joint Attention 实现双向信息交换

## 相关概念

- [[Mixture-of-Transformers]]
- [[Asymmetric Attention]]
- [[Action-DiT]]
- [[Video-DiT]]
- [[Transformer]]
