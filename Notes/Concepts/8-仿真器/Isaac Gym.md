---
type: concept
aliases: [IsaacGym, NVIDIA Isaac Gym]
---

# Isaac Gym

## 定义

NVIDIA 开发的高性能物理仿真平台，支持数千个并行 GPU 加速环境，专为机器人强化学习和模仿学习设计。

## 核心要点

1. **GPU 并行化**: 所有物理计算在 GPU 上执行，支持数千个同时仿真环境，大幅加速数据采集
2. **PhysX 引擎**: 基于 NVIDIA PhysX，提供刚体、关节、布料、软体等丰富物理模型
3. **与 Isaac Lab 的关系**: Isaac Gym 为早期版本，现已逐步被 [[Isaac Lab]] 取代（基于 Omniverse）

## 代表工作

- [[FurnitureVLA]]: 用于双臂家具装配的专家数据自动生成（运动规划 + 物理仿真）
- 众多 RL 操作研究的仿真基础

## 相关概念

- [[Isaac Lab]]: NVIDIA 新一代仿真平台，Isaac Gym 的后继者
- [[MuJoCo]]: 另一主流物理仿真引擎
- [[ManiSkill]]: 基于仿真的操作 benchmark
