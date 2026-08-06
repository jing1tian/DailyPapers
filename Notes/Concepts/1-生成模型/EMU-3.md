---
type: concept
aliases: [EMU3]
---

# EMU-3

## 定义
EMU-3：ByteDance 发布的多模态 token 生成模型，统一图像理解和生成，在 next-token prediction 范式下处理图像、视频和文本。

## 核心要点
1. 将图像/视频离散化为 token，用自回归模型统一生成和理解
2. 不依赖 diffusion，使用 VQVAE 离散化视觉内容
3. 作为 video VAE 压缩质量的比较对象

## 代表工作
- [[EmbodiedVAE]]: 与 EMU-3 的 VAE 压缩质量对比

## 相关概念
- [[SDXL]]
- [[VidTwin]]
