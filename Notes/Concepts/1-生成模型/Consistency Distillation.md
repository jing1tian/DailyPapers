---
type: concept
aliases: [一致性蒸馏, CD, LCM, Latent Consistency Model, Consistency Models]
---

# Consistency Distillation

## 定义

一致性蒸馏（Consistency Distillation，CD）是一种将多步扩散/流匹配模型蒸馏为单步或少步模型的训练范式，核心约束是：同一 ODE 轨迹上任意噪声级别的点经学生映射后，应到达相同的干净样本端点。

## 数学形式

**学生一致性函数**（流匹配参数化）：

$$
f_\theta(a_t, t) = a_t - t \cdot v_\theta(a_t, t)
$$

**CD 损失**（相邻点一致性约束）：

$$
\mathcal{L}_{CD} = \mathbb{E}\left[ \left\| f_\theta(a_t, t) - \text{sg}\left[f_{\bar{\theta}}(\hat{a}_{t-\Delta t}, t-\Delta t)\right] \right\|^2 \right]
$$

其中 $\hat{a}_{t-\Delta t}$ 是由教师 ODE solver 从 $a_t$ 推进一步得到的估计值，$\bar\theta$ 是 EMA 目标网络。

## 核心要点

1. **局部一致性约束**：CD 仅确保相邻时间步之间一致，不直接约束全局端点误差，可能导致"局部最优但全局漂移"。
2. **EMA 目标网络**：学生 $\theta$ 通过 EMA 维护慢速移动的目标 $\bar\theta$，避免训练塌陷。
3. **与 Progressive Distillation 的区别**：CD 不依赖教师网络逐步缩减步数，而是直接约束一致性。
4. **全局误差问题**：经验和理论均表明，CD 训练收敛后局部误差小，但到真实样本的全局误差（端点偏差）可能较大，是 [[TACD]] 等后续工作的动机。

## 代表工作

- [[Consistency Models]]（Song et al., 2023）：提出 CD 框架
- [[Flash-WAM]]：将 CD 应用于 WAM 联合蒸馏，发现模态差异问题
- [[RealtimeWAM]]：提出 [[TACD]] 解决 CD 的全局误差问题

## 相关概念

- [[Flow Matching]]
- [[TACD]]
- [[MeanFlow]]
- [[DMD]]
- [[LCM]]
