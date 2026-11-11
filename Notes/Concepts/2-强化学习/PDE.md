---
type: concept
aliases: [Prompt Driven Exploration]
---

# PDE

## 定义
用语言 prompt 变体作为 VLA 强化学习的探索空间，替代传统随机动作噪声，借助语言语义多样性诱导有意义的行为多样性。

## 核心要点
1. [[VLM]] 生成多样化语言 prompt 变体（同一意图的不同描述）
2. 每次 rollout 从 prompt 空间采样，驱动 [[PPO]] 探索
3. 语言诱导探索比随机动作噪声在语义层面更有效
4. 有真实机器人实验，MIT 出品（arXiv 2607.08837 v5）

## 代表工作
- [[PDE]]: 原始论文（MIT，2607.08837）

## 相关概念
- [[VLA]]
- [[PPO]]
- [[VLM]]
