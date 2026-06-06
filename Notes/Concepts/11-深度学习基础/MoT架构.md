---
type: concept
aliases: [MoT架构, Mixture-of-Transformer, MoT, 混合Transformer架构]
---

# MoT架构（Mixture-of-Transformer）

## 定义
Mixture-of-Transformer (MoT) 是一种将多个专用 Transformer 专家并联/串联的架构，每个专家负责处理特定模态或任务子目标，通过受控的注意力机制协同工作。

## 核心要点
1. **专家分工**: 不同专家专注于不同子任务（如语义理解、可供性生成、动作合成），避免单模型的多任务干扰
2. **单向因果注意力**: 信息流只能从前序专家到后续专家，防止信息泄漏和表示坍塌
3. **区别于标准 MoE**: MoT 是串行专家而非并行路由，每个专家都全量参与而非选择性激活

## 数学形式（AffordanceVLA 中的 UAA 注意力）

$$
h_{und} \rightarrow h_{gen} \rightarrow h_{act}
$$

- Affordance Expert 仅从 Understanding Expert 获取信息
- Action Expert 同时关注 Understanding 和 Affordance Expert

## 代表工作
- [[AffordanceVLA]]: 采用三专家 MoT（Understanding + Affordance Generation + Action）实现 VLA

## 相关概念
- [[混合专家架构]]
- [[视觉语言动作模型]]
- [[中间表示]]
