---
type: concept
aliases: [运动意图链式推理, Motion Chain-of-Thought, Motion Semantic Context]
---

# Motion-CoT

## 定义

Motion-CoT（Motion Chain-of-Thought）是一种在 [[World Action Model]] 中将高层运动意图显式编码为中间推理步骤的机制，使用一组可学习查询 token 从视觉-语言上下文中蒸馏运动语义，并将其作为 Chain-of-Thought 保留在推理空间中，用于引导精细动作轨迹的生成。

## 数学形式

Action Perceiver 提取 Motion-CoT：

$$
\mathbf{m} = \mathbf{O}[:, 1{:}K+1, :] \in \mathbb{R}^{K \times d}
$$

其中 $K=64$ 为查询 token 数量，$\mathbf{O}$ 为 Action Perceiver 输出，$\mathbf{m}$ 即运动语义上下文。

## 核心要点

1. **解耦作用**: 将高层运动语义（意图）从低层动作轨迹（运动）中分离，避免扩散噪声污染语义表征
2. **推理效率**: Motion-CoT token 在推理时直接可用，无需生成像素级未来帧，推理延迟降低 90%
3. **无监督聚类**: t-SNE 分析表明，Motion-CoT token 无需显式任务标签即可按动作类型自然聚类

## 代表工作

- [[PILOT]]: 首次提出 Motion-CoT，配合 [[Causally-Decoupled Attention]] 实现意图-轨迹解耦

## 相关概念

- [[Causally-Decoupled Attention]]
- [[Causal Dynamics Engine]]
- [[World Action Model]]
- [[Flow Matching]]
