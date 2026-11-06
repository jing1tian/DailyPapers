---
type: concept
aliases: [Vector Accelerated Motion Planning]
---

# VAMP

## 定义
向量化加速运动规划库：利用 SIMD 指令并行化碰撞检测和路径搜索，实现亚毫秒级 motion planning。

## 核心要点
1. 核心是向量化 (SIMD) 的碰撞检测，单次检测速度极快
2. 支持 PRM、RRT 等多种规划算法
3. [[cuRoboV2-Fast]] 对比方法之一，用于 benchmark CPU-based planning

## 相关概念
- [[CuRobo]]
- [[RRT]]
- [[cuRoboV2-Fast]]
