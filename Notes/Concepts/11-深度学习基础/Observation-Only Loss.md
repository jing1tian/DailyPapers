---
type: concept
aliases: [观测状态转移损失, 无意识学习损失]
---

# Observation-Only Loss

## 定义

Orca 无意识学习阶段的训练损失，无语言条件，直接从连续视频帧预测下一帧的 ViT 隐状态，权重最低（$\lambda=0.1$）。

## 数学形式

$$
\mathcal{L}_{\text{obs}} = \mathbb{E}\left[\ell_{\text{lat}}\left(\hat{v}^l_{t+1},\, v^l_{t+1}\right)\right]
$$

## 核心要点

1. 无标注，监督信号完全来自视频时序连续性
2. 权重设为 0.1 是为了避免大量无标注视频噪声主导训练
3. 消融实验显示此损失对文本生成有边际提升（三路联合时）

## 代表工作

- [[Orca]]: 三路预训练损失之一（$\lambda_{\text{obs}}=0.1$）

## 相关概念

- [[Unconscious Learning]]
- [[Latent Matching Loss]]
- [[Event-Conditioned Loss]]
- [[Pre-training Loss]]
