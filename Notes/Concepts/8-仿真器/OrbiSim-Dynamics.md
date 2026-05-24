---
type: concept
aliases: [对象中心动力学核心, Object-Centric Dynamics]
---

# OrbiSim-Dynamics

## 定义

OrbiSim 的物理动力学模块，将机器人和场景物体建模为离散 token，通过状态空间模型循环核心与 Transformer 耦合模块，实现对象中心的多实体物理状态预测。

## 核心要点

1. 每个实体（机械臂关节、操作物体）独立编码为 token，支持异构资产泛化
2. Transformer 耦合模块通过自注意力捕捉多实体交互
3. AdaLN 将物理常数（质量、摩擦）和控制信号注入动力学预测
4. 全可微分，奖励梯度可经由 Jacobian 链式传播回策略参数

## 数学形式

$$
\text{AdaLN}(u, c) = \gamma(c) \odot \text{LN}(u) + \beta(c)
$$

$$
\nabla_\theta J(\theta) = \frac{\partial R}{\partial x_T} \sum_{t=0}^{T-1} \left( \prod_{k=t+1}^{T-1} \frac{\partial x_{k+1}}{\partial x_k} \right) \frac{\partial x_{t+1}}{\partial a_t} \frac{\partial \pi_\theta(x_t)}{\partial \theta}
$$

## 代表工作

- [[OrbiSim]]: 提出 OrbiSim-Dynamics 的论文

## 相关概念

- [[状态空间模型]]
- [[Transformer]]
- [[AdaLN]]
- [[对象中心表示]]
- [[可微分仿真]]
