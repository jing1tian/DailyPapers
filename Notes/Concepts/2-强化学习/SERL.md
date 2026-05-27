---
type: concept
aliases: [Sample-Efficient Real-World RL, 样本高效真实RL]
---

# SERL

## 定义
**SERL**（Sample-Efficient Real-world Reinforcement Learning）是专为真实机器人在线强化学习设计的框架，通过高 UTD（Update-to-Data）比率和人工干预回放提升样本效率。

## 核心要点
1. 高 UTD 比率（远高于 1:1），每收集一条 transition 做多次梯度更新
2. Human-in-the-Loop（HIL）干预：人工接管避免危险行为并收集高质量 demo
3. 混合 offline demo + online 交互数据（类似 RLPD 策略）
4. 适合在真实机器人上从少量交互达到部署级成功率

## 代表工作
- [[SERL]]: Luo et al. (2024), UC Berkeley
- [[EXPO-FT]]: SERL 的 VLA finetuning 扩展（Stanford, 2026）

## 相关概念
- [[RLPD]]
- [[强化学习]]
- [[行为克隆]]
