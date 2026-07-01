---
type: concept
aliases: [动作条件化, 动作条件注入, Action-Conditioning]
---

# Action Conditioning

## 定义

以机器人执行动作作为条件信号，引导视觉预测模型或扩散模型生成动作驱动的场景变化，建立动作与视觉变化的因果映射。

## 数学形式

$$
p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k})
$$

常见注入方式：将动作序列经 MLP 编码为嵌入向量，加入至扩散模型的 timestep 嵌入中。

## 核心要点

1. **因果监督**: 动作提供比文本更精确的因果信号，迫使模型学习"动作→视觉变化"的物理映射
2. **注入方式**: 常见为 MLP 编码后与 timestep 嵌入相加，或通过 cross-attention 注入
3. **与文本条件对比**: 文本条件关注语义意图，动作条件关注精确运动轨迹；两者可结合

## 代表工作

- [[A2World]]: 以动作为条件大规模预训练 DiT 世界模型，学习可迁移动力学先验
- [[Action-Conditioned World Model]]: 通用动作条件世界模型范式

## 相关概念

- [[Action-Conditioned World Model]]
- [[Diffusion Model]]
- [[Diffusion Transformer (DiT)|DiT]]
