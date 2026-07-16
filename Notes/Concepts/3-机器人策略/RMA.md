---
type: concept
aliases: [Rapid Motor Adaptation, 快速运动自适应]
---

# RMA

## 定义
RMA（Rapid Motor Adaptation）是 UC Berkeley 提出的足式机器人 sim-to-real 自适应框架：基础策略在仿真中通过 privileged information 训练，部署时用轻量适配模块从历史观测在线估计环境参数，无需显式系统辨识。

## 数学形式
$$
a_t = \pi(o_t, \hat{e}_t), \quad \hat{e}_t = \phi(o_{t-H:t})
$$

$\hat{e}_t$ 为适配模块从历史观测估计的 extrinsics 向量，代替训练时使用的 privileged 环境参数。

## 核心要点
1. **两阶段训练**：Stage 1 用 privileged env parameters 训练基础策略；Stage 2 训练适配模块用历史观测预测这些参数
2. **Privileged learning**：训练和部署使用不同的信息，打破对特权信息的依赖
3. **在线自适应**：无需重新训练，适配模块运行于硬件上实时更新
4. **ANYmal 验证**：在现实中快速适应不同地面类型（草地、石块等）

## 代表工作
- [[RMA]]: Kumar et al., RSS 2021
- 足式运动 sim-to-real 的标杆之一
- [[HumanoidDribble]]: 人形足球运球借鉴 RMA 的 privileged representation

## 相关概念
- [[Human-to-Robot Retargeting]]
- [[Diffusion Policy]]
