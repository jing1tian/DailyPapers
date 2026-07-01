---
type: concept
aliases: [Vector Quantized VAE, 向量量化变分自编码器, VQVAE]
---

# VQ-VAE

## 定义

Vector Quantized Variational Autoencoder（向量量化变分自编码器）：将连续潜变量离散化为固定 codebook 中的 token 序列，实现图像/视频的离散表示学习，是 token-based 生成模型和世界模型的核心组件。

## 数学形式

给定输入 $x$，编码器输出连续嵌入 $z_e$，通过最近邻查找量化：

$$
z_q = \arg\min_{c_k \in \mathcal{C}} \| z_e - c_k \|_2
$$

解码器以 $z_q$ 为输入重建 $\hat{x}$。训练目标：

$$
\mathcal{L} = \| x - \hat{x} \|^2 + \| \text{sg}(z_e) - z_q \|^2 + \beta \| z_e - \text{sg}(z_q) \|^2
$$

其中 $\text{sg}$ 为 stop-gradient，$\beta$ 为承诺损失系数。

## 核心要点

1. **离散化**: 将连续特征映射到有限 codebook（通常 $K = 512 \sim 8192$ 个向量）
2. **直通梯度**: 量化操作不可微，用 straight-through estimator 传递梯度
3. **紧凑表示**: 图像可压缩为短序列 token，支持自回归生成
4. **Codebook 更新**: 通常用 EMA 或重置（reset）更新 codebook 向量，防止坍缩

## 代表工作

- [[ITC]]: 在 Atari 100K 实验中使用 VQ-VAE tokenizer（Codebook 大小 4096）
- [[IRIS]]: 最早将 VQ-VAE 用于 Transformer 世界模型的工作
- [[SA-VLA]]: 将 VQ-VAE 扩展为状态感知动作 tokenizer，通过 MLP Adapter 预测状态条件缩放因子

## 相关概念

- [[最近邻查找]]: VQ-VAE 量化步骤的核心操作
- [[FSQ]]: VQ-VAE 的简化变体（Finite Scalar Quantization）
- [[VQGAN]]: 结合 GAN 训练的 VQ-VAE
- [[MAGVIT]]: 高效视频 VQ-VAE
