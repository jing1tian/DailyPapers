---
type: concept
aliases: [MAGVIT, MAGVIT-v2, Masked Generative Video Transformer]
---

# MAGVIT

## 定义
Google 提出的掩码生成视频 Transformer，使用 VQVAE 将视频离散化为 token，然后用双向 Transformer 做掩码预测，实现高效视频生成和理解。

## 数学形式
$$
p(x) = \prod_{t \in \mathcal{M}} p(x_t | x_{\bar{\mathcal{M}}}, c)
$$
在掩码位置 $\mathcal{M}$ 条件于未掩码 token $x_{\bar{\mathcal{M}}}$ 和条件信号 $c$ 做联合预测。

## 核心要点
1. 3D VQ-VAE 将视频时空 patch 离散化为整数 token 序列
2. 掩码预测比自回归生成更高效（并行解码）
3. MAGVIT-v2 引入 lookup-free quantization 提升重建质量
4. 可用于视频生成、预测、分类（统一框架）

## 代表工作
- Yu et al. (2022): "MAGVIT: Masked Generative Video Transformer"

## 相关概念
- [[V-JEPA]]
- [[JEPA]]
- [[DiT]]
