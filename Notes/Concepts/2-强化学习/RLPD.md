---
type: concept
aliases: [RL with Prior Data, 先验数据强化学习]
---

# RLPD

## 定义
**RLPD**（Reinforcement Learning with Prior Data）是一种将 offline 先验数据与 online RL 融合的框架，利用历史 demo 缩短 RL 探索期，实现更高样本效率。

## 核心要点
1. 混合 replay buffer：offline 先验 demo + online 交互数据
2. 采样时按固定比例从两个 buffer 中采样（如 50:50）
3. 高 UTD（Update-to-Data ratio）训练，每步收集后多次更新
4. 不需要 BC 预训练，直接用 SAC/TD3 等 off-policy 算法

## 代表工作
- [[RLPD]]: Ball et al. (2023), Berkeley
- [[SERL]]: 在真实机器人中实例化 RLPD 思路
- [[EXPO-FT]]: 将 RLPD 用于 VLA finetuning

## 相关概念
- [[SERL]]
- [[强化学习]]
- [[行为克隆]]
