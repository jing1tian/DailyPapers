---
type: concept
aliases: [UniZero]
---

# UniZero

## 定义
在 MuZero 框架上统一多任务、多环境的 model-based RL 方法，通过共享隐空间世界模型支持跨游戏/跨任务泛化。

## 核心要点
1. MuZero 的多任务扩展，共享表示网络和动态网络跨环境
2. 用 transformer 替换 LSTM 做时序建模，提升长程规划能力
3. 在 Atari 多任务 benchmark 上达到新 SOTA
4. [[COMET]] 以 UniZero 为主要对比 baseline

## 代表工作
- [[COMET]]：与 UniZero 在 ManiSkill 和 VizDoom 上对比

## 相关概念
- [[MuZero]]
- [[MCTS]]
- [[COMET]]
