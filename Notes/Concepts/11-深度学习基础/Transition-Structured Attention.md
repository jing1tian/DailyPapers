---
type: concept
aliases: [转移结构化注意力, 转移注意力]
---

# Transition-Structured Attention

## 定义

在无标注视频预训练中，Action-DiT 的注意力被设计为同时关注干净的前帧潜变量 $z_{k-1}$ 和**加噪**的后帧潜变量 $z_{k,\tau_v}$，使动作 token 能从帧间差异中学习转移先验，而加噪设计防止直接复制目标帧。

## 数学形式

**Action-DiT 注意力**（预训练阶段）：

$$
H_{A,k} = \operatorname{Attn}\left(Q_{A,k},\; [K_{V,k-1};\, K_{V,k,\tau_v};\, K_{A,k}],\; [V_{V,k-1};\, V_{V,k,\tau_v};\, V_{A,k}]\right)
$$

**Video-DiT 注意力**（加噪后帧，预训练阶段）：

$$
H_{V,k,\tau_v} = \operatorname{Attn}\left(Q_{V,k,\tau_v},\; [K_{V,k-1};\, K_{V,k,\tau_v};\, K_{A,k}],\; [V_{V,k-1};\, V_{V,k,\tau_v};\, V_{A,k}]\right)
$$

其中 $z_{k,\tau_v} = (1-\tau_v)z_k + \tau_v\epsilon_k^v$ 为加噪后帧。

## 核心要点

1. **转移信息提取**：前帧 $z_{k-1}$ 提供上下文，加噪后帧 $z_{k,\tau_v}$ 提供模糊的未来信息，动作 token 需从两者差异推断转移
2. **加噪防止直接复制**：纯净后帧 $z_k$ 包含完整信息，不加噪则 Action-DiT 可绕过动作路径直接复制（Proposition A.2 理论证明加噪必要性）
3. **视频损失反传动作路径**：Video-DiT 的去噪损失梯度通过 $H_{V,k,\tau_v}$ 中对 $K_{A,k}$ 的注意力流回 Action-DiT，直接优化动作参数

## 代表工作

- [[NAVA-WAM]]: 提出该注意力设计，实现在无动作标签视频上的 Action-DiT 原生预训练

## 相关概念

- [[Joint Attention]]
- [[Asymmetric Attention]]
- [[Flow Matching]]
- [[Action-DiT]]
- [[Video-DiT]]
