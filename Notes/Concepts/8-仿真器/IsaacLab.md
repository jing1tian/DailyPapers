---
type: concept
aliases: [Isaac Lab, NVIDIA Isaac Lab, 机器人强化学习仿真框架]
---

# IsaacLab

## 定义
NVIDIA 开发的基于 Isaac Sim 的机器人学习框架，提供 GPU 并行化的物理仿真环境，专门用于强化学习和模仿学习。

## 核心要点
1. 基于 NVIDIA PhysX 物理引擎，支持数千个并行环境同时仿真（GPU 加速）
2. 统一的任务接口（Task API）支持操作、运动、导航等多类机器人任务
3. 与 Isaac Sim 无缝集成，支持高质量渲染和传感器仿真
4. 是 Isaac Gym 的继任者，API 更现代化

## 代表工作
- 广泛用于 locomotion、manipulation 的 sim-to-real 研究

## 相关概念
- [[MuJoCo]]
- [[SAPIEN]]
- [[Isaac Gym]]
