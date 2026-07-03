---
type: concept
aliases: [TD-MPC2, Temporal Difference Model Predictive Control v2]
---

# TD-MPC2

## 定义

TD-MPC2 是一种基于模型的强化学习框架，将时序差分学习（TD）与模型预测控制（MPC）结合，在隐空间中同时学习动力学模型与价值函数，通过规划实现高效策略执行。

## 核心要点

1. 在 latent space 中学习 encoder + dynamics + reward + value 网络
2. 用 MPPI 或 CEM 在预测轨迹上做 MPC 规划
3. 支持连续控制任务，推理时无需大量 rollout
4. 相比 TD-MPC v1 扩展到更多任务并改进了架构（LayerNorm、FFN 等）

## 代表工作

- [[Valdi]]: 把 dynamics model 替换为 diffusion model，同时估计 value

## 相关概念

- [[MPC]]
- [[DIAMOND]]
- [[DreamerV3]]
