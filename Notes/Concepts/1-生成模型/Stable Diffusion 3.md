---
type: concept
aliases: [SD3, Stable Diffusion 3, SD 3]
---

# Stable Diffusion 3

## 定义

Stable Diffusion 3（SD3）是 Stability AI 发布的多模态扩散 Transformer 图像生成模型，采用 MMDiT（Multi-Modal Diffusion Transformer）架构，引入[[流匹配]]训练目标替代传统 DDPM，并配备高压缩比的 16 通道 VAE（将图像压缩到 1/8 空间分辨率），在文本-图像生成质量上显著提升。

## 数学形式

SD3 VAE 编码过程：

$$
z = \mathcal{E}_\eta(I) \in \mathbb{R}^{H/8 \times W/8 \times 16}
$$

解码：

$$
\hat{I} = \mathcal{D}_\eta(z)
$$

生成模型基于[[流匹配]]目标：

$$
\mathcal{L} = \mathbb{E}_{x^0, x^1, \tau}\left[\left\| (x^1 - x^0) - f_\theta(x^\tau, \tau, c) \right\|_2^2\right]
$$

## 核心要点

1. **高压缩比 VAE**：16 通道、8× 空间压缩，比 SD1/2 的 4 通道 VAE 语义表达能力更强
2. **MMDiT 架构**：文本与图像 token 在独立流中处理后通过交叉注意力融合，优于单流注意力
3. **流匹配训练**：比 DDPM 收敛更快，支持更少 NFE 推理
4. **作为视觉编码器**：SD3 VAE 可作为冻结的强大视觉特征提取器使用，保留丰富的视觉语义先验

## 代表工作

- SD3 论文（Esser et al., 2024）：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
- [[WEAVER]]: 冻结 SD3 VAE 编码器用于机器人操作世界模型，将多视角观测编码为语义丰富的潜在 token，避免从头训练带来的 OOD 鲁棒性问题

## 相关概念

- [[VAE]]
- [[流匹配]]
- [[Rectified Flow]]
- [[潜扩散模型]]
- [[Video Diffusion Model]]
