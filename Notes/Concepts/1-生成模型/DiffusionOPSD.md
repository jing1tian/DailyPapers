---
type: concept
aliases: [On-Policy Self-Distillation for Diffusion, 扩散模型在线策略自蒸馏]
---

# DiffusionOPSD

## 定义
On-Policy Self-Distillation in Diffusion Models：将图像级 RL reward 梯度转化为去噪步骤显式监督目标的框架，使扩散模型能够从任务特定 reward 信号中学习，而不仅依赖端点奖励。

## 数学形式
$$\mathcal{L}_{OPSD} = \mathbb{E}_{q \sim \pi_\theta}\left[\sum_t \|D_\theta(x_t, t) - \hat{x}_0^+(x_t)\|^2\right]$$

其中 $\hat{x}_0^+$ 是由 reward 梯度构造的正向目标。

## 核心要点
1. 冻结行为策略生成轨迹，提供 query states 和 anchor
2. Reward 梯度围绕每个 anchor 构造有界正负目标
3. 可训练策略通过离线监督拟合这些目标，绕过传统 RLHF 的不稳定性
4. On-policy 采样确保学习分布与生成分布一致

## 代表工作
- [[DiffusionOPSD]] (2608.24646): 原始方法提出（完整笔记见 [[DiffusionOPSD]]）

## 相关概念
- [[Diffusion Policy]]
- [[GRPO]]
- [[Flow Matching]]
