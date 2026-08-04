---
type: concept
aliases: [VLA RL, VLA Reinforcement Learning]
---

# VLA-RL

## 定义

VLA-RL 是早期将强化学习引入视觉-语言-动作模型的工作，探索在线 RL 对 VLA 策略的提升效果。

## 核心要点

1. 使用在线 RL 对预训练 VLA 模型进行微调
2. 在 LIBERO benchmark 上平均成功率 81.0%，低于基于 SFT 的 OpenVLA-OFT（89.2%）
3. 奠定 VLA 在线 RL 适应的基础范式

## 代表工作

- [[Agentic-VLA]]: 在 LIBERO 上超越 VLA-RL（97.8% vs 81.0%）
- [[WCM]]: 通过世界预测 critic 提升 VLA-RL 中价值估计质量，149 个任务上达到 SOTA

## 相关概念

- [[EVOLVE-VLA]]
- [[SimpleVLA-RL]]
- [[GRPO]]
- [[视觉语言动作模型]]
