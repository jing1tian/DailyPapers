---
type: concept
aliases: [SD, Stable Diffusion, Latent Diffusion Model, LDM]
---

# SD

## 定义

Stable Diffusion（SD）是 Stability AI 基于 Latent Diffusion Model（LDM）训练的开源文本-图像扩散模型，是目前最广泛使用的图像生成基础模型之一。

## 核心要点

1. 在 VAE 压缩的 latent space 中进行扩散过程，降低计算成本
2. 用 CLIP text encoder 做文本条件注入（cross-attention）
3. UNet 架构做去噪
4. 开源权重（v1.4/v1.5/v2.x/SDXL/SD3 等多个版本）
5. 是 [[DreamBooth]]、[[ESD]]、[[SPM]] 等大量个性化/安全方法的基础

## 相关概念

- [[DreamBooth]]
- [[ESD]]
- [[DIAMOND]]
