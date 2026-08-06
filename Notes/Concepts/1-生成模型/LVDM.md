---
type: concept
aliases: [Latent Video Diffusion Model]
---

# LVDM

## 定义
Latent Video Diffusion Model：在 latent 空间而非 pixel 空间进行视频扩散的生成模型，大幅降低计算成本。

## 核心要点
1. 先用 VAE 将视频压缩到 latent 空间
2. 在 latent 空间运行扩散过程（DDPM/Flow-Matching）
3. 解码时还原到 pixel 空间

## 代表工作
- [[DriftWorld]]: 与 LVDM 类方法对比，提出 one-step 替代方案

## 相关概念
- [[VDM]]
- [[CogVideoX]]
