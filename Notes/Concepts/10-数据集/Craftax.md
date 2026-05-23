---
type: concept
aliases: [Craftax-classic, Craftax benchmark, NetHack-like RL benchmark]
---

# Craftax

## 定义

Craftax 是一个基于 JAX 实现的高效 2D 开放世界游戏 RL 基准，灵感来自 Minecraft，具有程序化生成的地图、复杂任务层次和随机动力学，是评测世界模型长程规划能力的主要 benchmark。

## 核心要点

1. **两个版本**:
   - **Craftax-classic**: 较小屏幕，任务较简单，标准版本（63×63 像素）
   - **Craftax**: 更大屏幕、更多物品和敌人，更高难度
2. **程序化生成**: 每次 episode 的地图、物品位置随机，需要真正的泛化能力
3. **任务层次**: 需要按顺序完成多个子目标（如收集资源 → 制造工具 → 对抗敌人）
4. **随机生物**: 环境中有随机移动的生物（creatures），对世界模型的 token 预测构成挑战
5. **评估指标**: Return（回报，0-100%）和 Score（任务完成里程碑，0-100%）

## 代表工作

- [[ITC]]: Craftax-classic 达到 72.5% Return / 35.6% Score（1M 步 SOTA）
- [[IRIS]]: 早期 Transformer 世界模型在此 benchmark 的测试
- [[DreamerV3]]: 53.2% Return，作为主要对比基线

## 相关概念

- [[World Model]]
- [[IRIS]]
- [[DreamerV3]]
- [[ITC]]
