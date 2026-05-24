---
type: concept
aliases: [Vid2World, Video-to-World]
---

# Vid2World

## 定义
基于视频扩散的世界模型，专注于从视频输入预测未来帧，作为机器人操作世界模型的基线方法之一。

## 核心要点
1. 纯视觉生成方法，无显式物理状态表示。
2. 在短时预测（10步）上有一定竞争力，但长时预测（100步）质量显著退化。
3. 缺乏资产级可控性，难以响应物理参数变化。

## 代表工作
- [[OrbiSim]]: 在 Robosuite Push 上对比，Vid2World FVD=1750.1，TrajErr=0.675，OrbiSim 分别为 533.9 和 0.447

## 相关概念
- [[World Model]]
- [[AdaWorld]]
- [[DreamerV3]]
