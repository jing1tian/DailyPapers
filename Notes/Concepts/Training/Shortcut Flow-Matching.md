---
type: concept
category: Training
tags: [generative, flow-matching, diffusion, distillation]
created: 2026-06-27
---

# Shortcut Flow-Matching

## 定义

Shortcut Models（Frans et al., 2025，*One Step Diffusion via Shortcut Models*）是一种支持**可变步长**的 [[Flow Matching]] 训练目标：网络除了条件于噪声水平 $\sigma$，还额外条件于目标步长 $d$，使同一个模型既能以传统多步 Euler 积分采样，也能以极少数步（甚至一步）完成采样，而无需像传统扩散蒸馏那样训练独立的教师-学生模型对。

## 数学形式

训练时混合两种监督信号：

1. **经验一步回归项**（最细步长 $d$）：直接回归 $(x_1 - z)/(1-\sigma)$ 形式的速度场目标。
2. **自洽 bootstrap 项**（占比 $\rho_{\text{self}}$）：在 `no_grad` 下用两个更粗的半步（step+1）预测的平均速度作为当前步速度场的 stop-gradient 目标，将两次粗步预测"蒸馏"进一次细步预测：

$$
v_\theta(x, \sigma, d) \leftarrow \text{stopgrad}\left[\tfrac{1}{2}\left(v_\theta(x, \sigma, 2d) + v_\theta(x', \sigma', 2d)\right)\right]
$$

推理时的 Euler 积分：

$$
b = \frac{\hat{x}_1 - z}{1-\sigma}, \qquad z \leftarrow z + b \cdot d
$$

## 核心要点

1. **步长作为条件变量**：噪声水平 $\sigma$ 与步长 $d$ 一起编码为条件 token，单一网络支持任意步数采样（4~8 步即可，而非传统扩散的几十至上百步）
2. **自蒸馏而非师生蒸馏**：自洽损失项让模型把自己粗粒度的多步预测蒸馏为细粒度的少步预测，无需冻结教师模型
3. **适配自回归世界模型**：相比传统 diffusion forcing，少步采样大幅降低世界模型逐帧 rollout 的推理延迟，使闭环规划（如 [[MPC]] / [[CEM]]）可行

## 代表工作

- [[MMBench2]]: 在 [[Dreamer-v4|Dreamer 4]] 风格的 [[Block Causal Attention|block-causal]] 动态模型中使用 shortcut flow-matching，仅需 4~8 个 Euler 子步完成下一帧采样，并将子步间预测的振荡幅度（flow instability $u_f$）用作幻觉检测信号

## 相关概念

- [[Flow Matching]]
- [[Conditional Flow Matching]]
- [[Diffusion Transformer (DiT)]]
