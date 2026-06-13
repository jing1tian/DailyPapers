---
type: concept
aliases: [Vision Tokenizer, 视觉 Tokenizer]
---

# ViTok

## 定义
基于 ViT 架构的视觉 tokenizer，将图像/视频压缩为离散或连续 token，通常以像素级重建为优化目标，常作为视频生成模型（如 WAM、世界模型）的视觉编码器。

## 数学形式
给定输入图像 $x \in \mathbb{R}^{H \times W \times C}$，ViTok 将其分块后编码为潜向量序列：
$$z = \text{ViT-Encoder}(\text{patchify}(x)) \in \mathbb{R}^{N \times d}$$

## 核心要点
1. **重建取向**：优化目标是像素级重建（MSE/LPIPS），倾向于保留视觉细节
2. **控制效率低**：保留了大量与动作控制无关的纹理/背景信息
3. **与 WAM 的关系**：被 [[RepWAM]] 等工作批评为不适合 WAM 场景，应换用表示对齐目标

## 代表工作
- 原始 ViTok 用于视频生成预训练
- [[RepWAM]]: 提出 RepViTok 替代 ViTok 用于世界动作模型

## 相关概念
- [[RepViTok]]
- [[Visual Tokenizer]]
- [[VQ-VAE]]
