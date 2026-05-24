---
type: concept
aliases: [IsaacLab, Isaac Lab, NVIDIA Isaac Lab]
---

# Isaac Lab

## 定义
NVIDIA 开发的基于 Isaac Sim 的模块化机器人强化学习框架，提供 GPU 并行仿真环境，支持多种机器人任务（操作、运动、抓取等）。

## 核心要点
1. 基于 NVIDIA Omniverse / PhysX 物理引擎，支持高精度刚体和关节模拟。
2. 内置大量标准任务（Stack、Lift、Reach 等），易于 RL 基准测试。
3. 支持 Isaac Lab-IL（模仿学习）流程，可结合视觉和状态输入训练策略。
4. GPU 并行仿真使大规模 RL 训练成为可能。

## 代表工作
- [[OrbiSim]]: 用 Isaac Lab Stack（225 步叠放任务）评测长时视频预测能力

## 相关概念
- [[MuJoCo]]
- [[World Model]]
- [[Reinforcement Learning]]
