---
type: concept
aliases: [OrbiSim-Dynamics, OrbiSim Dynamics, 对象中心物理动力学]
---

# OrbiSim-Dynamics

## 定义
OrbiSim 框架的物理动力学核心模块，基于对象中心表示的循环神经网络，自回归预测物理状态（位置、速度、关节角），支持端到端梯度反传。

## 数学形式

$$
\hat{x}_t = f_\phi^{dyn}(x_{0:t-1}, a_{t-1}, \bar{x})
$$

$$
z_t = f_\phi^{enc}(h_t, x_t), \quad h_t = f_\phi^{rec}(h_{t-1}, e_t)
$$

$$
e_t = f_\phi^{cp}(z_{t-1}, a_{t-1}, \bar{x})
$$

## 核心要点
1. 每个场景实体（机器人关节、操作物体）编码为独立 token。
2. **Transformer 耦合模块**：通过多头自注意力捕捉多实体交互，AdaLN 注入物理常数和控制信号。
3. **循环模块**（SSM/RNN）：维持跨时间步的隐状态，记忆历史动力学。
4. 完全可微分，支持奖励梯度通过 Jacobian 链式传播到策略参数。

## 代表工作
- [[OrbiSim]]: 完整系统描述

## 相关概念
- [[对象中心表示]]
- [[状态空间模型]]
- [[多头自注意力]]
- [[AdaLN]]
- [[可微分策略优化]]
