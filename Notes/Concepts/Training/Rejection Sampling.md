---
type: concept
aliases: [拒绝采样, RS, Rejection Sampling Fine-tuning, RFT]
---

# Rejection Sampling

## 定义

拒绝采样（Rejection Sampling）是一种数据筛选策略，通过让模型对提示生成多个候选输出，然后根据质量标准（如正确性、格式）过滤保留高质量样本，用于构建 SFT 训练数据。

## 数学形式

$$
\mathcal{D}_{SFT} = \{(x_i, y_i) \mid y_i \in \text{sample}(f_\theta, x_i), \; \text{quality}(y_i) \geq \tau\}
$$

其中 $\tau$ 为质量阈值，$f_\theta$ 为当前模型，$x_i$ 为提示。

## 核心要点

1. 与 RLHF 相比，拒绝采样更简单、无需奖励模型，适合快速数据构建
2. 保留率（retention rate）反映数据质量，过高说明标准过松，过低影响数据量
3. 在 LLM 训练中常作为 SFT 阶段的数据准备步骤，再接 RL 精调
4. Qwen-AgentWorld 中全局保留率为 69.2%，各域 59.5%～86.5%

## 代表工作

- [[QwenAgentWorld]]: 在 SFT 阶段对 7 域多轮轨迹进行拒绝采样，保留率 69.2%

## 相关概念

- [[GSPO]]: 拒绝采样后的 RL 精调算法
- [[RLHF]]: 同类数据驱动的偏好对齐方法
- [[Language World Model]]: 应用场景
