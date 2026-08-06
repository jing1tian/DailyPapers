---
type: concept
aliases: [Latent World-Action Model]
---

# LaWAM

## 定义
Latent World-Action Model：在 latent 空间而非 pixel 空间进行世界模型预测和动作生成的 WAM 变体。

## 核心要点
1. 用 VAE 将视觉观测压缩到 latent 空间
2. 所有预测和动作生成在 latent 空间进行
3. 相比 pixel-level WAM 计算效率更高

## 代表工作
- [[LiLa-WAM]]: 轻量 latent reasoning WAM for manipulation
- [[EmbodiedVAE]]: 专为 manipulation 设计的解耦视频 VAE

## 相关概念
- [[WAM]]
- [[LiLa-WAM]]
