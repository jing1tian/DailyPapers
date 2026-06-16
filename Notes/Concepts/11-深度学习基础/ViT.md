---
type: concept
aliases: [Vision Transformer, ViT, 视觉Transformer]
---

# ViT (Vision Transformer)

## 定义
将图像分割为 patch 序列，用标准 Transformer（self-attention）直接处理视觉输入的骨干网络，由 Dosovitskiy et al. (2020) 提出。

## 数学形式
输入图像 $I \in \mathbb{R}^{H \times W \times C}$ 切成 $N$ 个 patch，线性嵌入后加位置编码：
$$z_0 = [x_{class}; x_1^p E; \ldots; x_N^p E] + E_{pos}$$
$$z_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$

## 核心要点
1. 无归纳偏置（卷积的平移等变性），需要大量数据或预训练
2. [[DINOv2]]、SigLIP、CLIP-ViT 是常用预训练 ViT 变体
3. VLA 系统几乎都用 ViT 做视觉 encoder（SigLIP-ViT-400M 等）
4. [[ContactWorld]]、[[COMET]] 等工作均以 ViT 为特征提取骨干

## 代表工作
- Dosovitskiy et al. (2020): 原始 ViT 论文
- [[DINOv2]]：自监督 ViT 预训练

## 相关概念
- [[DINOv2]]
- [[SigLIP]]
- [[CLIP]]
- [[Action Expert]]
