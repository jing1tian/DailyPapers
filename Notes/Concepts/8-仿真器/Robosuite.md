---
type: concept
aliases: [Robosuite, robosuite]
---

# Robosuite

## 定义
基于 MuJoCo 的模块化机器人仿真框架，提供标准化的机器人操作任务套件，广泛用于模仿学习和强化学习基准测试。

## 核心要点
1. 支持多种机器人平台（UR5e、Panda、Sawyer 等）和末端执行器。
2. 内置 Push、Lift、Stack、NutAssembly 等经典操作任务。
3. 支持随机化物理参数（摩擦、质量）进行鲁棒性测试。
4. 常用于世界模型和策略学习研究的标准评测环境。

## 代表工作
- [[OrbiSim]]: 在 Robosuite Push 任务上评测稀疏奖励 RL，成功率 42.71% vs DreamerV3 的 25%

## 相关概念
- [[MuJoCo]]
- [[Isaac Lab]]
- [[Reinforcement Learning]]
