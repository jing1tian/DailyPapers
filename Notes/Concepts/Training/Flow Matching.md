---
concept: Flow Matching
category: Training
tags: [diffusion, flow-matching, generative]
created: 2026-05-09
---

# Flow Matching

## 定义

Flow Matching (FM) 是一类生成模型训练目标。给定真实样本 $x_1$ 与噪声 $x_0 = \varepsilon$，构造直线插值 $x_\tau = \tau x_1 + (1-\tau) x_0$，让网络 $u_\theta$ 学习速度场 $u_\theta(x_\tau, \tau) \approx x_1 - x_0$。

## 标准损失（Conditional FM）

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau, \varepsilon, x_1} \left\| u_\theta(x_\tau, \tau) - (x_1 - \varepsilon) \right\|_2^2
$$

## 推理（欧拉积分）

$$
x_{\tau + \Delta\tau} = x_\tau + \Delta\tau \cdot u_\theta(x_\tau, \tau), \quad x_0 = \varepsilon
$$

## 与 Diffusion 的对比

| 维度 | Diffusion (DDPM) | Flow Matching |
|------|------------------|---------------|
| 路径 | 弯曲（按 noise schedule） | 直线 |
| 目标 | 噪声预测 | 速度场 |
| 推理步数 | 通常更多 | 通常更少 |
| 实现 | 复杂 schedule | 极简 |

## 代表工作

- [[Pi0]] (2024): 首个用 FM 训练 VLA。
- [[Pi05]]: 增强版。
- [[RLDX-1]]: 在 action 与 physics 双流上都用 FM。
- [[MolmoAct2]] (2026): Action Expert 用 FM，预训练 K=4、微调 K=8 的多样本损失。
- [[GEM-4D]] (2026): 双流流匹配框架，视频 DiT 与几何 DiT 各自用 FM 训练，几何蒸馏损失通过中间特征梯度回传到视频主干。
- [[SC3-Eval]] (2026): 在 [[Cosmos3]] backbone 上沿用 [[Rectified Flow|rectified-flow]] 形式化的 FM 目标，正向动力学、跨视角补全、逆向动力学三种训练模式共享同一套 FM 损失，仅通过哪些 token 加噪来区分模式。
- [[Qwen-RobotManip]] (2026): 在 Flow-Matching DiT 动作专家中使用带逐维掩码的 FM 损失，支持跨 15 种机器人平台的异构动作空间联合训练。
- [[InternVLA-A1.5]] (2025): 以 Beta(1.5,1.0) 采样插值时间步的流匹配损失预测连续动作速度场，推理时用 Euler 积分去噪。

## 相关概念

- [[Diffusion Model]]
- [[Action Chunking]]
- [[Euler Integration]]
