---
type: concept
aliases: [世界模型, WM]
---

# World Model

## 定义
学习环境动力学的可预测模型 $p(s_{t+1}\mid s_t, a_t)$（或其 latent 形式），用于规划、想象式 rollout 与决策。

## 核心要点
1. **像素重建型**（Dreamer 系）：在观测空间重建，容量浪费。
2. **Latent / JEPA 型**（LeWM、PLDM、DINO-WM）：在 latent 空间预测，规划开销低。
3. 配合 [[Model Predictive Control|MPC]] / [[Cross-Entropy Method|CEM]] 做规划。
4. 对 embodied AI / 机器人是核心组件。

## 代表工作
- [[LeWM]]：端到端 JEPA 世界模型，仅 2 项 loss
- [[DINO-WM]]：foundation feature + 规划器
- [[PLDM]]：端到端但 7 项 loss
- [[Dreamer]] 系列：像素重建型
- [[WAM-Survey]]：首篇系统综述将世界模型与动作生成统一为 WAM 范式
- [[WMRobotSurvey]]：NTU/Berkeley 团队综述，从策略集成、RL 仿真器、视频生成三轴系统梳理机器人学习世界模型

## 相关概念
- [[JEPA]]
- [[Model Predictive Control]]
- [[Cross-Entropy Method]]
- [[Representation Collapse]]
