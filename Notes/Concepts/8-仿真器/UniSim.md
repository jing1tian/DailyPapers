---
type: concept
aliases: [Universal Simulator]
---

# UniSim

## 定义
基于真实世界视频训练的通用视觉世界模型，可作为机器人策略训练的视频仿真器，通过 action-conditioned 视频生成模拟多种交互场景。

## 核心要点
1. 从无标签视频训练，无需 action label
2. 可 condition 在 language / action 上生成视频 rollout
3. 用于 model-based RL 或数据扩充
4. 与 [[LAM]]（latent action model）方向紧密关联

## 代表工作
- [[CD-LAM]]：将 UniSim 作为 baseline 对比 latent action world model 方法

## 相关概念
- [[LAM]]
- [[AdaWorld]]
