---
type: concept
aliases: [ConRFT, Consistency Policy Reinforced Fine-Tuning]
---

# ConRFT

## 定义

ConRFT（A Reinforced Fine-tuning Method for VLA Models via Consistency Policy）是一种结合行为克隆与 Q 学习的 VLA 强化微调方法，使用一致性策略（consistency policy）作为动作生成模型，在离线预训练后进行在线 RL 微调。

## 核心要点

1. **一致性策略 backbone**：用一致性模型替代标准扩散策略，加速动作采样
2. **离线-在线两阶段**：先用 BC + Q 学习联合训练离线策略，再在线微调
3. **可选人类在环（HIL）变体**：支持人类干预加速收敛，论文对比时常用 "no HIL" 版本作为纯自动化 baseline
4. **样本效率**：相比纯 BC 或纯 offline RL，显著提升真实机器人任务成功率

## 局限性

- 在更复杂的多任务/工具使用场景下成功率仍有限（如 PullCubeTool 类任务）
- 对探索数据质量未做显式过滤，与 [[VGPD]] 等显式过滤机制相比效率较低

## 代表工作

- [[FORCE]]: 将 ConRFT（no HIL）作为最强 RL 微调 baseline 进行对比，在 ManiSkill 任务上平均成功率超越约 10+ 个百分点

## 相关概念

- [[强化学习]]
- [[行为克隆]]
- [[Diffusion Policy]]
- [[Cal-QL]]
