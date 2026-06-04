---
type: concept
aliases: [Diffusion-based Model-Augmented RL]
---

# DIAMOND

## 定义
基于扩散模型的世界模型强化学习方法，使用扩散过程建模环境动力学，实现高质量视觉世界模型 RL。

## 核心要点
1. 用扩散模型代替传统 RSSM 建模环境转移
2. 生成更逼真的未来帧，提升 imagination rollout 质量
3. 在 Atari 等视觉 RL benchmark 上有强结果

## 代表工作
- DIAMOND: Diffusion for World Model-Based RL (2024)

## 相关概念
- [[DreamerV3]]
- [[RSSM]]
- [[扩散模型]]
