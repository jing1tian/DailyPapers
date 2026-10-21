---
type: concept
aliases: [RGB变分自编码器, 视频VAE, Video VAE, Latent Video Encoder]
---

# RGB VAE

## 定义
RGB VAE 是用于将 RGB 视频帧编码到低维潜在空间的变分自编码器，通常来自预训练的视频生成模型（如 [[Wan]]、Stable Video Diffusion），冻结后用作下游策略学习的特征提取器。

## 数学形式

编码：$z = \mathcal{E}(x), \quad x \in \mathbb{R}^{T \times H \times W \times 3}$

解码：$\hat{x} = \mathcal{D}(z)$

潜在空间维度：通常 $z \in \mathbb{R}^{T' \times h \times w \times C}$，$C$ 为潜在通道数。

## 核心要点

1. **预训练冻结使用**: 通常直接使用视频生成模型的 VAE，不在策略训练中微调
2. **时间压缩**: 将视频序列压缩到更低帧率的潜在表示
3. **空间压缩**: 通常 8× 或更高的空间下采样
4. **与深度 VAE 配对**: 在 PrismWAM 等方法中，RGB VAE 与深度 VAE 并行使用

## 代表工作
- [[PrismWAM]]: 冻结 Wan RGB VAE 编码观测历史和目标帧，配合深度 VAE 实现解耦状态表示

## 相关概念
- [[Wan]]
- [[Flow Matching]]
- [[Depth Prediction]]
