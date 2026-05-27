---
type: concept
aliases: [Diffusion Steering RL, Steering Diffusion Policy in Latent Space]
---

# DSRL

## 定义

DSRL（Diffusion Steering Reinforcement Learning，或 Steering Diffusion Policy in Latent Space）是一种通过在扩散策略的潜空间中进行引导优化来实现 RL 微调的方法，避免直接修改扩散模型参数，而是在推理过程中引导采样方向。

## 核心要点

1. **潜空间引导**: 在扩散去噪过程中注入 RL 梯度信号，引导采样向高奖励区域
2. **参数冻结**: 不修改预训练扩散策略的参数，保留预训练先验
3. **推理时优化**: 优化发生在推理阶段而非训练阶段，降低训练开销
4. **先验约束**: 受限于原始扩散策略的先验分布，难以探索远离训练分布的行为

## 局限性

- 探索受限于初始策略的先验分布，无法发现超出预训练数据分布的最优行为
- 在需要大幅修正预训练策略的精密任务上表现受限
- 潜空间优化可能与实际动作空间的语义不对齐

## 代表工作

- [[EXPO-FT]]: 与 DSRL 对比，展示直接 RL 微调（而非潜空间引导）的优越性

## 相关概念

- [[扩散策略]]
- [[强化学习]]
- [[VLA]]
