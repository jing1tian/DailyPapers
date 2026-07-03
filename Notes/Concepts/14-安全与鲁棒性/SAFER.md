---
type: concept
aliases: [SAFER]
---

# SAFER

## 定义

SAFER 是一种基于控制障碍函数（CBF）的安全过滤器方法，用于在三维场景表示（如 3DGS）上为机器人/无人机运动提供安全约束保证。

## 核心要点

1. 使用 [[CBF]] + QP 框架构建安全约束
2. 在场景几何表示上构建障碍函数
3. 是 [[FastBridge]] 等工作的对比基线
4. 处理 model-based realization gap（模型假设与真实动力学的差距）

## 相关概念

- [[CBF]]
- [[FastBridge]]
