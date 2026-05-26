---
type: concept
aliases: [EVOLVE-VLA, Evolving VLA]
---

# EVOLVE-VLA

## 定义

EVOLVE-VLA 是一种基于强化学习的 VLA 在线适应方法，通过与环境交互进行策略优化，是 Agentic-VLA 的主要对比基线。

## 核心要点

1. 使用 RL 方法对预训练 VLA 进行在线微调
2. 需要约 1,680 iterations（53.8k rollouts）才能收敛到 90% 成功率（LIBERO-Long）
3. 缺乏自适应奖励机制、结构化探索引导和跨任务知识共享

## 代表工作

- [[Agentic-VLA]]: 在样本效率（2.4×）和最终性能（+2.0%）上超越 EVOLVE-VLA

## 相关概念

- [[OpenVLA-OFT]]
- [[GRPO]]
- [[视觉语言动作模型]]
