---
type: concept
aliases: [Multimodal VAE]
---

# MVAE

## 定义
Multimodal VAE：将多个异质模态（图像、文本、动作等）编码到统一 latent 空间的变分自编码器架构。

## 数学形式
$$q(z | x_1, x_2, ..., x_n) = \prod_i q_i(z | x_i)$$

## 核心要点
1. 用乘积专家（Product of Experts）融合多模态信息
2. 共享 latent 空间实现跨模态对齐
3. 在 VLA 中用于统一动作和视觉目标的表征

## 代表工作
- [[UVT]]: 用 MVAE 统一多个视觉运动目标的监督信号

## 相关概念
- [[UVT]]
- [[VAE]]
