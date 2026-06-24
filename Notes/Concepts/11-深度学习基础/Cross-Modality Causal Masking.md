---
type: concept
aliases: [跨模态因果掩码, Cross-Modal Causal Mask, 模态非对称注意力掩码]
---

# Cross-Modality Causal Masking

## 定义

一种在 [[Mixture-of-Transformers|MoT]] 多专家架构中使用的**非对称**注意力掩码策略：不同模态的 token 被赋予不同的可见性规则（而非按时间帧分块），使信息单向流动——通常是让低维、高敏感度的模态（如动作）以高维、信息丰富的模态（如视觉）为条件，而不允许反向干扰。

## 数学形式

设视觉 token 集合 $\mathcal{V}$、动作 token 集合 $\mathcal{A}$、价值 token 集合 $\mathcal{P}$，注意力掩码 $M$ 定义为：

$$
M(i, j) = \begin{cases}
1 & i, j \in \mathcal{V} \\
1 & i \in \mathcal{A},\ j \in \mathcal{V} \cup \mathcal{A} \\
1 & i \in \mathcal{P},\ j \in \mathcal{V} \cup \mathcal{A} \cup \mathcal{P} \\
0 & \text{otherwise}
\end{cases}
$$

即视觉 token 仅自注意力；动作 token 可 attend 视觉与动作；价值 token 可 attend 全部三者。

## 核心要点

1. **单向信息流**: 高保真度生成模态（视觉）不被低维模态（动作/价值）的梯度或注意力污染，保护其生成质量
2. **条件式生成**: 低维模态以高维模态预测的未来上下文为条件，实现"动作以世界动态为条件"
3. **与 [[Block Causal Attention]] 的区别**: Block Causal Attention 按**时间帧**分块、保留时间因果性；Cross-Modality Causal Masking 按**模态类型**分块，约束的是模态间的信息可见性而非时间顺序，两者可叠加使用
4. **位置编码对齐**: 常配合不同模态分支使用不同的 [[RoPE|RoPE 缩放因子]]，实现跨模态的时间步对齐

## 代表工作

- [[MV-WAM]]: 视觉 token 仅自注意力，动作 token attend 视觉+动作，价值 token attend 三者全部，用于在 [[Mixture-of-Transformers|MoT]] 双专家间实现非对称信息流动

## 相关概念

- [[Mixture-of-Transformers]]: Cross-Modality Causal Masking 通常在该架构的共享注意力层中实现
- [[Block Causal Attention]]: 另一种因果掩码范式，按时间帧而非模态分块
- [[RoPE]]: 常与跨模态掩码配合，解决跨模态时间对齐问题
- [[World Action Model]]: 该掩码策略的主要应用场景
