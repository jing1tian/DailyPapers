---
type: concept
aliases: [整流流, RF, Rectified Flow Matching]
---

# Rectified Flow

## 定义

一种基于[[Flow Matching]]的生成模型训练框架，通过学习从噪声到数据的直线（整流）路径来训练速度场网络，具有训练稳定、推理步数少的优点。

## 数学形式

插值轨迹（线性插值）：
$$z_t = (1-t)z_0 + t\epsilon, \quad t \sim \mathcal{U}(0, 1)$$

训练目标（最小化速度场预测误差）：
$$\mathcal{L}_\text{RF} = \mathbb{E}_{t,z_0,\epsilon} \left\| v_\theta(z_t, t, c) - (\epsilon - z_0) \right\|^2_2$$

## 核心要点

1. 使用**线性插值**在数据 $z_0$ 和噪声 $\epsilon$ 之间构造训练轨迹，目标速度为常数 $(\epsilon - z_0)$
2. 相比 DDPM 的曲线路径，整流路径更直，推理时可用更少 ODE 求解步数
3. 与 [[Flow Matching]] 等价，但强调"整流"（直化）路径的几何直觉
4. 被 Stable Diffusion 3、[[Cosmos-Predict2.5]] 等模型采用

## 代表工作

- [[OSCAR]]: 将 Rectified Flow 作为动作条件视频世界模型的训练目标
- [[Cosmos-Predict2.5]]: 使用 Rectified Flow 训练的大规模视频扩散模型

## 相关概念

- [[Flow Matching]]
- [[Diffusion Transformer]]
- [[DDIM]]
