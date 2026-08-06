---
type: concept
aliases: [Temporal Difference Model Predictive Control, TD-MPC2]
---

# TD-MPC

## 定义
TD-MPC：结合时序差分（TD）学习和模型预测控制（MPC）的强化学习方法，在 latent 世界模型中进行 planning。

## 数学形式
$$V(s_t) = \max_{a_t} \left[ r(s_t, a_t) + \gamma V(s_{t+1}) \right]$$

## 核心要点
1. 学习隐式 latent 世界模型，而非 pixel-level 预测
2. 在 latent 空间 rollout 进行 MPC 规划
3. TD-MPC2 扩展到多任务和大规模设定

## 代表工作
- [[Quo-Vadis-WM]]: 引用 TD-MPC 作为 latent WM + planning 的代表方法

## 相关概念
- [[DreamerV3]]
- [[MuJoCo]]
