---
type: concept
aliases: [Stable Diffusion XL]
---

# SDXL

## 定义
Stable Diffusion XL：Stability AI 发布的大规模文本-图像扩散模型，在 SDXL 基础架构上使用两阶段生成（base + refiner）。

## 核心要点
1. 1024×1024 分辨率生成，显著高于 SD 1.5/2.x
2. 双 text encoder（CLIP-L + OpenCLIP bigG）
3. 作为视频/操控世界模型的生成基础

## 代表工作
- [[EmbodiedVAE]]: 与 SDXL VAE 比较视频压缩质量

## 相关概念
- [[FLUX.md]]
- [[ControlNet]]
