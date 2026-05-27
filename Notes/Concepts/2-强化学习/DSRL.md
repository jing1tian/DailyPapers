---
type: concept
aliases: [Diffusion Steering RL, 扩散策略引导强化学习]
---

# DSRL

## 定义

DSRL（Diffusion Policy Steering via Reinforcement Learning）是一种通过在潜空间中优化引导信号来改进扩散策略的方法，无需直接修改扩散模型的参数即可注入奖励信号。

## 核心要点

1. **潜空间引导**: 在扩散去噪过程中，通过优化潜变量注入 Q 值/奖励引导
2. **冻结预训练模型**: 不修改扩散策略参数，保留预训练分布
3. **局限**: 优化被约束在原始策略先验的流形内，难以探索先验分布之外的行为
4. **与 EXPO-FT 对比**: EXPO-FT 通过直接微调 VLA 参数实现更大范围的策略改进

## 代表工作

- [[EXPO-FT]]: 作为对比基线，通过直接参数微调克服了 DSRL 受限于先验分布的局限

## 相关概念

- [[扩散策略]]
- [[强化学习]]
- [[EXPO]]
- [[Actor-Critic]]
