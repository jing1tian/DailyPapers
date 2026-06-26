---
type: concept
aliases: [Diffusion Policy, DP]
---

# Diffusion Policy

## 定义
用 [[Diffusion Model|条件扩散模型]] 直接回归视觉运动策略 $\pi(a_{t:t+H} \mid o_t)$，输出动作 chunk。Chi et al., RSS 2023。

## 核心要点
1. 多模态动作分布友好，远胜 MSE 回归 / MDN。
2. 一般搭配 [[Action Chunking|动作块]] 与 receding-horizon 执行。
3. 在 [[Push-T]]、Square、Tool-Hang 等基准上是强基线。

## 代表工作
- Diffusion Policy (Chi 2023)
- 3D Diffusion Policy / DP3
- 各类 [[VLA]] 工作把扩散头作为 action expert
- [[SVP-IL]]: 在 DP 的 U-Net 噪声预测器前融合空间视觉提示特征，提升语言条件下多物体消歧能力

## 相关概念
- [[Diffusion Model]]
- [[Action Chunking]]
- [[VLA]]
