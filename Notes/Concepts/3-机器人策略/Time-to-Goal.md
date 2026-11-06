---
type: concept
aliases: [Time to Goal metric, 时序目标度量]
---

# Time-to-Goal

## 定义
用于 JEPA world model 规划的时序度量方法：训练一个 temporal metric head 预测给定 latent state 距离 goal 所需步数，替代 latent 向量距离来指导 [[CEM]] 规划。

## 核心要点
1. 核心动机：latent 空间距离不保证时序单调性——更接近 goal 的 state 不一定 latent 距离更小
2. 分离 temporal metric 和 spatial representation，各自优化
3. 在 [[LeWorldModel]] (JEPA) 框架上实现，[[AntMaze]] 上验证
4. 配合 [[CEM]] 做 trajectory optimization

## 相关概念
- [[LeWM]]
- [[JEPA]]
- [[CEM]]
- [[LeWorldModel]]
