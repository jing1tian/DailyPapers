---
type: concept
aliases: [Deep Compression Autoencoder, 深压缩自编码器]
---

# DCAE

## 定义
高压缩率视觉自编码器，将图像/视频压缩到极低空间分辨率的 latent space，以支持更长序列的扩散模型训练和推理。

## 数学形式
压缩比 $r = H/h$，DCAE 实现极高的空间压缩比（如 $r=32$），相比标准 VAE（$r=8$）大幅降低 token 数：
$$N_{token} = \frac{H \times W}{r^2}$$

## 核心要点
1. 与标准 [[VAE]] 相比，DCAE 追求更高压缩率而非重建质量的极致
2. 高压缩率使 [[DiT]] 能在更短的 token 序列上操作，支持分钟级长视频生成
3. 重建质量损失被扩散模型的生成能力补偿
4. [[SANA-Video]] 使用 DCAE 实现 720×1280 长视频的高效生成

## 代表工作
- [[SANA-Video]]（NVIDIA, 2025）: 使用 DCAE + Linear DiT 实现高效视频生成

## 相关概念
- [[VAE]]: 标准变分自编码器，压缩率较低
- [[LDM]]: 潜扩散模型，依赖 VAE/DCAE 压缩
- [[DiT]]: 扩散变换器，DCAE 的下游模型
