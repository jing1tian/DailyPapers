---
type: concept
aliases: [Time Optimal trajectory optimization]
---

# TimeOpt

## 定义
[[cuRoboV2-Fast]] 中集成的轨迹时间最优化模块，在 B-spline 轨迹优化后对时间参数重新计划，使轨迹在满足速度/加速度/力矩约束下尽可能快。

## 核心要点
1. 与 B-spline 轨迹优化分阶段：先优化路径形状，再优化时间参数化
2. 结合 differentiable inverse dynamics 计算力矩约束
3. GPU 并行化实现

## 相关概念
- [[TOPPRA]]
- [[cuRoboV2-Fast]]
- [[CuRobo]]
