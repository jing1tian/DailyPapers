---
type: concept
aliases: [Vector Quantized GAN, Taming Transformers]
---

# VQGAN

## 定义
向量量化 GAN，将图像压缩到离散 codebook 表示，用 Transformer 在 codebook 空间做自回归生成，同时用对抗训练保证重建质量，是图像/视频离散 token 化的重要基础。

## 数学形式
$$z_q = \arg\min_{c_k \in \mathcal{C}} \|z_e - c_k\|_2, \quad \mathcal{L} = \mathcal{L}_\text{rec} + \mathcal{L}_\text{GAN} + \|\text{sg}(z_e) - z_q\|^2$$

## 核心要点
1. Encoder 将图像映射到连续 latent，量化为最近 codebook 向量
2. GAN 判别器约束重建质量，比纯 VAE 重建更锐利
3. 量化后 Transformer 在 token 序列上做自回归生成
4. LVDrive 用 VQGAN 将场景图像做 latent encoding 作为辅助监督

## 代表工作
- [[VQGAN]]：Esser et al. 2021，Taming Transformers，CVPR 2021
- [[LVDrive]]：自动驾驶 VLA 用 VQGAN 做 latent 视觉辅助任务

## 相关概念
- [[VAE]]
- [[Diffusion Model]]
- [[潜扩散模型]]
