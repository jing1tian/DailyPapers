---
type: concept
aliases: [Classifier-Free Guidance for Reinforcement Learning]
---

# CFGRL

## 定义
将 Classifier-Free Guidance 思路引入 diffusion policy 的 RL 训练，通过 guidance scale 在行为克隆和 Q-value 最大化之间插值。

## 核心要点
1. 无需单独的 classifier，直接通过 conditional/unconditional score 的差值施加 reward guidance
2. 在 diffusion sampling 阶段注入 RL 信号

## 代表工作
- [[DIPOLE]]: 与 CFGRL 对比的方法

## 相关概念
- [[Classifier-Free Guidance (CFG)]]
- [[DIPOLE]]
