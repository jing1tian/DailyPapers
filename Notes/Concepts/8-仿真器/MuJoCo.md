---
type: concept
aliases: [MuJoCo, Multi-Joint dynamics with Contact]
---

# MuJoCo

## 定义
MuJoCo（Multi-Joint dynamics with Contact）是一款高性能物理仿真引擎，专为机器人控制、强化学习和生物力学研究设计，支持快速、精确的刚体和关节动力学模拟。

## 核心要点
1. 基于广义坐标系（generalized coordinates）建模，比笛卡尔坐标系更适合关节链系统
2. 支持接触力的精确建模，包括摩擦、碰撞响应
3. 被广泛用于 DeepMind、OpenAI 等机构的强化学习基准测试
4. 2021 年被 DeepMind 收购后开源

## 代表工作
- [[BISON]]：在 MuJoCo 中验证双层策略长序列规划
- [[DreamerV3]]：在 MuJoCo 控制任务上测试世界模型

## 相关概念
- [[Reinforcement Learning]]
- [[World Model]]
