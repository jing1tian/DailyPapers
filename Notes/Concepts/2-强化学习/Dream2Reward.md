---
type: concept
aliases: [Dream2Reward, Transition-Alignment Reward]
---

# Dream2Reward

## 定义
一种从正向演示数据学习 language-conditioned 奖励模型的方法，通过对比每步 transition 与成功演示的 latent 轨迹对齐程度来提供密集奖励，解决 manipulation RL 的稀疏奖励问题。

## 核心要点
1. 成功 latent 轨迹编码：将正向 demo 编码为 latent transition 序列
2. Transition-Alignment Reward：每步奖励来自当前 transition 与成功 latent 轨迹的对齐度
3. Q-Guided Flow（QGF）用于离线 RL 中的策略学习
4. 在 LIBERO 在线 RL 和真实机器人离线 RL 上均有实验

## 代表工作
- Zhang et al., 2026 — [[Dream2Reward]] (arXiv 2608.18787, Sun Yat-sen University)

## 相关概念
- [[ROBOMETER]]
- [[Robo-Dopamine]]
- [[RoboReward]]
- [[LIBERO]]
