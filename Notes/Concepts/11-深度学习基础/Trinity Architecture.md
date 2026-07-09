---
type: concept
aliases: [三位一体架构, 三元架构, Physical AGI 架构]
---

# Trinity Architecture

## 定义

**Trinity Architecture**（三位一体架构）是通向 Physical AGI 的系统架构框架，由世界模型、价值/目标模型、动作/策略模型三个互补组件构成闭环智能体系统。

## 数学形式

三个组件的形式化角色：

$$
\text{World Model: } \mathbf{s}_{t+1} \sim W_\theta(\cdot \mid \mathbf{s}_t, \mathbf{a}_t)
$$

$$
\text{Value Model: } V_\psi(\mathbf{s}) = \mathbb{E}\left[\sum_{k=0}^\infty \gamma^k r_{t+k} \mid \mathbf{s}_t = \mathbf{s}\right]
$$

$$
\text{Policy Model: } \mathbf{a}_t = \pi_\phi(\mathbf{s}_t, \mathbf{z}_t, V_\psi)
$$

## 核心要点

1. **世界模型**（World Model）: 状态理解与转移预测，提供"想象环境"
2. **价值/目标模型**（Value Model）: 对目标状态或未来轨迹赋予价值，定义"什么是好的"
3. **动作/策略模型**（Action/Policy Model）: 基于当前信念和价值估计输出控制指令
4. 三者形成**闭环自改善**：世界模型支持无限制探索 → 价值模型评估 → 策略迭代优化
5. 区别于纯行为模仿（Behavior Cloning）：通过理解驱动而非仅模仿数据

## 代表工作

- [[WorldModelRoadmap]]: 提出 Trinity Architecture 作为 Physical AGI 路径

## 相关概念

- [[World Model]]
- [[World Action Model]]
- [[MBRL]]
- [[Chain-of-Imagination]]
