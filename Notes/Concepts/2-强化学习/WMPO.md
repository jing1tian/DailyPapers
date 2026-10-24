---
type: concept
aliases: [World Model Policy Optimization, 世界模型策略优化]
---

# WMPO（World Model Policy Optimization）

## 定义

WMPO 是一类用于 VLA 强化学习后训练的双阶段框架：先以像素 MSE 训练视频世界模型，再在世界模型提供的虚拟 rollout 中运行 GRPO 更新策略，避免昂贵的真实仿真器交互。

## 数学形式

**阶段1：世界模型像素拟合**

$$
\mathcal{L}_{\mathrm{MSE}}(\phi) = \mathbb{E}_{(o_{\leq t},a_t,o_{t+1})\sim\mathcal{D}}\!\left[\|\hat{o}_{t+1}(\phi)-o_{t+1}\|_2^2\right]
$$

**阶段2：虚拟 GRPO 策略更新**

$$
\hat{J}_{\mathrm{GRPO}}(\pi_\theta;\mathcal{W}_\phi)=\mathbb{E}_{\tau\sim\mathcal{W}_\phi}\!\left[\frac{1}{H}\sum_{t=0}^{H-1}\hat{A}(o_t,a_t)\log\pi_\theta(a_t\mid o_{\le t})\right]
$$

## 核心要点

1. **解耦训练**: 世界模型和策略交替更新，世界模型目标（像素 MSE）与策略优化目标完全解耦
2. **优势**: 大幅减少真实仿真器调用次数，虚拟 rollout 可并行生成
3. **目标错配问题**: [[DRILL]] 证明低像素 MSE 不等于良好的策略梯度信号——17% 更高 MSE 的世界模型可产生 +13 点更高成功率

## 代表工作

- [[DRILL]]: 提出 WMPO 目标错配问题，用双层目标替换像素 MSE，在 5 仿真器上均值成功率提升 13.1 点

## 相关概念

- [[GRPO]]
- [[世界模型]]
- [[双层优化]]
- [[DRILL]]
