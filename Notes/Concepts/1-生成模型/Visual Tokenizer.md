---
type: concept
aliases: [视觉 Tokenizer, 视觉编码器, Video Tokenizer, Image Tokenizer]
---

# Visual Tokenizer

## 定义

Visual Tokenizer 是将连续的图像或视频帧压缩为紧凑离散/连续 token 序列的编码模块，使 Transformer 等序列模型能高效处理视觉信息，是多模态生成模型的基础组件。

## 数学形式

$$
z = \mathcal{E}(x), \quad x' = \mathcal{D}(z)
$$

其中 $\mathcal{E}$ 为编码器，$\mathcal{D}$ 为解码器，$z$ 为压缩后的 token 表示，$x$ 为原始图像/视频输入。

## 核心要点

1. **空间压缩**: 通过卷积下采样将空间分辨率压缩 8x 或 16x，大幅降低 token 数量
2. **时序压缩**: 对视频在时间维度额外压缩 4x 或 8x（因果时序卷积），总压缩倍数可达 2048x
3. **因果设计**: 采用因果时序卷积（causal temporal conv）+ 因果时序注意力，保持视频帧的自然时序顺序，支持流式生成
4. **离散 vs 连续**: 离散 tokenizer（如 VQ-VAE）输出离散 token id，适合 AR 模型；连续 tokenizer（如 VAE）保留潜变量，适合扩散模型
5. **图像-视频统一**: 高质量 tokenizer 以统一架构处理图像（视为单帧视频）和视频

## 代表工作

- [[Cosmos3]]: NVIDIA Cosmos Tokenizer 采用因果时序卷积+注意力，空间 8x/16x + 时序 4x/8x，总压缩最高 2048x
- [[VAE]]: 标准连续 tokenizer，用于潜扩散模型（LDM）
- [[VQ-VAE]]: 离散 tokenizer，输出离散 token id，供 AR 模型使用
- [[VQGAN]]: 基于 GAN 训练的高质量离散 tokenizer

## 相关概念

- [[VAE]]
- [[VQ-VAE]]
- [[VQGAN]]
- [[Diffusion Model]]
- [[Autoregressive Transformer]]
- [[Efficient Video Sampling]]
