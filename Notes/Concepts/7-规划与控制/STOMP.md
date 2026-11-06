---
type: concept
aliases: [Stochastic Trajectory Optimization for Motion Planning]
---

# STOMP

## 定义
随机轨迹优化运动规划算法：通过对轨迹添加随机噪声生成候选，根据代价函数（碰撞、平滑度）加权平均更新，迭代收敛到低代价轨迹。

## 核心要点
1. 不需要代价函数梯度，适合不可微约束（如碰撞检测）
2. 通过采样 + 加权平均而非梯度下降优化
3. 比 CHOMP 在某些非凸代价场景更鲁棒
4. [[cuRoboV2-Fast]] 对比方法之一

## 相关概念
- [[CHOMP]]
- [[RRT]]
- [[MPPI]]
- [[cuRoboV2-Fast]]
