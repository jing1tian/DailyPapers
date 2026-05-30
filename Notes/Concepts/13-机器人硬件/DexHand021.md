---
type: concept
aliases: [DexHand021, 灵巧手021]
---

# DexHand021

## 定义
一款 21 自由度（DoF）五指灵巧机器人手，用于高难度 dexterous manipulation 研究的真实硬件平台。

## 核心要点
1. 21 个自由度提供近人类级别的手指灵活性
2. 高维控制空间（21 DoF）使得传统 BC 策略训练困难，需要 RL 后训练补偿
3. [[BORA]] 的真实硬件实验平台，验证了 offline RL + online residual 适应框架

## 代表工作
- [[BORA]]: 在 DexHand021 上验证 offline RL + online PPO 适应

## 相关概念
- [[强化学习]] — 灵巧手通常需要 RL 进行精细控制
- [[VLA（视觉-语言-动作模型）]]
