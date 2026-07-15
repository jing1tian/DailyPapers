---
type: concept
aliases: [BAR, 块状自回归, Block Autoregression, 分块自回归]
---

# Block-wise Autoregression (BAR)

## 定义

**块状自回归**（Block-wise Autoregression, BAR）是一种推理加速策略：将 token 序列组织为**语义块**，块**内**并行生成（无因果约束），块**间**保持自回归顺序，从而在维持生成质量的同时大幅降低延迟。

## 数学形式

设 token 序列被分为 $K$ 个块 $\{B_1, B_2, \ldots, B_K\}$，每块包含 $M$ 个 token：

$$
p(T_{1:KM}) = \prod_{k=1}^{K} p(B_k \mid B_1, \ldots, B_{k-1})
$$

其中块内并行：$p(B_k) = \prod_{m=1}^{M} p(t_{(k-1)M+m} \mid B_{<k})$（块内不施加因果掩码）

## 核心要点

1. **块内并行**: 同一语义块内的 token 同时生成，减少自回归步数
2. **块间因果**: 块之间仍维持自回归顺序（后一块以前一块为条件），保持生成一致性
3. **推理加速**: Lumo-2 中，4-8 token 块大小实现 **2.71×** 推理加速（RTX 5090 + vLLM）
4. **语义分组**: 将 action token 按语义组织（Lumo-2 中为 8 个语义动作组），使块内 token 天然具有并行化合理性

## 代表工作

- [[Lumo-2]]: 将 action token 分为 8 个语义块，通过 BAR 实现 2.71× 推理加速

## 相关概念

- [[Block Causal Attention]]: BAR 通过特殊设计的块状因果注意力掩码实现
- [[Autoregressive Policy]]: 标准自回归策略，BAR 是其推理加速变体
- [[Action Chunking]]: 动作块输出，BAR 在块级别进行并行化
