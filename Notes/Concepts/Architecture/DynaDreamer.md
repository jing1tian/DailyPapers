---
type: concept
aliases: [DynaDreamer, Ego-Dynamics Augmented World Model]
---

# DynaDreamer

## 定义
在 DreamerV3 框架基础上引入 ODE 自我动力学（Ego-Dynamics）模型的自动驾驶世界模型，支持 zero-shot 跨底盘迁移。

## 数学形式
$$s_{t+1}^{\text{ego}} = \text{ODE}(s_t^{\text{ego}}, a_t; \phi_{\text{chassis}})$$
$$z_{t+1} = \text{WM}(z_t, s_{t+1}^{\text{ego}}; \theta)$$

其中 $\phi_{\text{chassis}}$ 为底盘参数（轴距、轮胎力系数），可独立替换以实现跨底盘迁移。

## 核心要点
1. ODE 模型预测自车状态（速度、横摆角），通过 AdaLN 注入世界模型
2. 跨底盘 zero-shot：仅更新 ODE 底盘参数，无需重训完整 WM
3. 基于 [[DreamerV3]] 骨干；BEV 表征
4. 局限：ODE 线性轮胎假设在极端驾驶条件下失效

## 代表工作
- DynaDreamer（2607.13410, NTU）: 驾驶 WM 跨底盘 zero-shot 迁移

## 相关概念
- [[DreamerV3]]
- [[Action-Conditioned World Model]]
- [[AdaLN]]
