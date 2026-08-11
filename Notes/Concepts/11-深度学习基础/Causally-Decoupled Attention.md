---
type: concept
aliases: [因果解耦注意力, Causal Decoupled Attention, Asymmetric Attention]
---

# Causally-Decoupled Attention

## 定义

Causally-Decoupled Attention 是一种非对称注意力机制，在同一 Action Perceiver 中对不同类型 token 实施不同的信息可见性规则：非动作（语义查询）token 只能访问上下文，而动作 token 额外访问运动语义；从而防止扩散噪声通过注意力反向污染语义表征。

## 数学形式

**非动作语义 token**（Motion-CoT 查询 $\mathbf{Z}$）仅见上下文 $\mathbf{c}$：

$$
\mathbf{Z} \leftarrow \mathbf{Z} + \text{Attn}(\mathbf{Z},\; [\mathbf{c};\mathbf{Z}])
$$

**动作 token**（$\mathbf{A}^\tau$，含扩散噪声）额外访问语义查询：

$$
\mathbf{A}^\tau \leftarrow \mathbf{A}^\tau + \text{Attn}(\mathbf{A}^\tau,\; [\mathbf{c};\mathbf{Z};\mathbf{A}^\tau])
$$

## 核心要点

1. **单向信息流**: 语义 → 动作（不可逆），防止噪声反向污染
2. **保留 CoT 完整性**: 语义 token 计算时不接触含噪动作，确保 [[Motion-CoT]] 语义纯净
3. **消融验证**: 去掉解耦注意力（退化为对称注意力）导致 LIBERO -0.8pp，RoboCasa -1.5pp

## 代表工作

- [[PILOT]]: 首次提出，与 [[Motion-CoT]] 和 [[Causal Dynamics Engine]] 配合使用

## 相关概念

- [[Motion-CoT]]
- [[World Action Model]]
- [[Flow Matching]]
- [[Transformer]]
