---
type: concept
aliases: [Reinforcement Learning from Human Feedback, 基于人类反馈的强化学习]
---

# RLHF（基于人类反馈的强化学习）

## 定义

利用人类偏好标注训练奖励模型，再通过强化学习（通常为 PPO）优化语言模型或策略以最大化该奖励，从而与人类偏好对齐。

## 核心要点

1. 核心流程：收集人类偏好 → 训练奖励模型 → RL 优化策略
2. InstructGPT、ChatGPT 等的核心对齐技术
3. 相比 [[SFT]]，RLHF 能更好地优化复杂偏好目标
4. 在机器人领域的类比：从成功/失败 rollout 学习奖励，[[COAST]] 则直接从中提取引导方向

## 相关概念

- [[SFT]]
- [[Reinforcement Learning]]
- [[Activation Steering]]
