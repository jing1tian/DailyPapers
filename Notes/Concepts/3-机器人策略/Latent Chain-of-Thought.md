---
type: concept
aliases: [潜在思维链, Latent CoT, 隐式推理链]
---

# Latent Chain-of-Thought

## 定义

将 Chain-of-Thought 推理从文本空间迁移到连续潜在表示空间，以结构化专家 Token 作为感知到规划之间的中间推理步骤，保留文字无法表达的连续时空信息。

## 数学形式

$$
p_\theta(A_{t+1:t+T} \mid o_t, c_t, \mathcal{Z}), \quad \mathcal{Z} = \{z_{sem},\, z_{geo},\, z_{dyn},\, z_{traj}\}
$$

其中 $\mathcal{Z}$ 为多类专家 Token 构成的潜在中间推理集合。

## 核心要点

1. **对比文本 CoT**: 文本推理步骤丢失连续几何和动态信息，Latent CoT 保留稠密时空表示
2. **多专家结构化**: 不同专家 Token 对齐不同世界知识来源（语义/几何/动态/轨迹），各有分工
3. **推理时主动参与**: 专家 Token 不仅在训练时提供监督，在推理时也参与规划器的条件生成

## 代表工作

- [[CoWorld-VLA]]: 首次将多专家潜在思维链用于自动驾驶 VLA，NAVSIM v1 PDMS 90.0

## 相关概念

- [[Chain-of-Thought Reasoning]]: 文本空间版本
- [[World Model]]: 提供潜在推理的世界先验
- [[HMEF]]: 将 Latent CoT 专家 Token 融合为轨迹的规划器
- [[Multi-Expert Training]]: 实现多专家 Token 对齐的训练方法
