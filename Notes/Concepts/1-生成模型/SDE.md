---
type: concept
aliases: [Stochastic Differential Equation, 随机微分方程, Score SDE, Diffusion SDE]
---

# SDE（随机微分方程）

## 定义

随机微分方程（SDE）是包含随机噪声项的微分方程，常用于生成模型中描述**从噪声到数据的随机扩散/反扩散过程**，或用于为确定性 ODE 引入可控随机性以支持策略探索。

## 数学形式

生成模型中的通用前向 SDE（加噪过程）：

$$
dx = f(x, t)\,dt + g(t)\,dw
$$

反向 SDE（去噪/生成过程，Anderson 1982）：

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}
$$

SimWAM 中用于 Flow Matching 探索的 SDE（将 ODE 扩展为 SDE）：

$$
\begin{aligned}
dx_\tau &= \left[v_\theta(x_\tau,\tau) + \frac{\sigma^2_\tau}{2\tau}\left(x_\tau + (1-\tau)v_\theta(x_\tau,\tau)\right)\right]d\tau + \sigma_\tau\,dw \\
\sigma_\tau &= a\sqrt{\frac{\tau}{1-\tau}}
\end{aligned}
$$

该形式保持与原 ODE 相同的边缘分布，但引入随机性，支持强化学习探索。

## 核心要点

1. **ODE vs SDE**: 确定性 ODE 每次推理轨迹相同；SDE 每次采样均不同，适合多样性探索。
2. **边缘分布保持**: 设计良好的 SDE 可在不改变数据分布的情况下引入可控噪声。
3. **Score Function 联系**: 反向 SDE 需要 score function $\nabla_x \log p_t(x)$；[[FlowMatching]] 可视为 SDE 的简化特例（线性插值路径）。
4. **RL 应用**: 在 RL 策略优化中，SDE 采样可生成多条候选轨迹供奖励评估（如 [[GRPO]]）。

## 代表工作

- [[SimWAM]]: 将 Flow Matching ODE 扩展为 SDE，为 GRPO RL 提供轨迹多样性探索，最终 PDMS 91.5。
- Score-Based Generative Models (Song et al. 2021): 基于 SDE 的生成模型统一框架。
- [[DDIM]]: 确定性 ODE 采样（SDE 噪声强度趋零的特例）。

## 相关概念

- [[FlowMatching]]: 基于 ODE 的生成路径，SDE 是其随机扩展
- [[GRPO]]: SimWAM 中使用 SDE 为 GRPO 提供轨迹探索
- [[Rectified Flow]]: 线性 ODE 路径，可扩展为 SDE
