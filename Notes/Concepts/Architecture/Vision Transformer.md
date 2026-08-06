---
type: concept
aliases: [ViT, Vision Transformer, 视觉Transformer, 视觉变换器]
---

# Vision Transformer (ViT)

## 定义
把图像切成固定大小 patch、加位置编码，送入标准 Transformer 编码器的视觉骨干。Dosovitskiy et al., ICLR 2021。

## 数学形式
$$
z_0 = [x_{cls}; \mathbf{E}\,x_{p_1}; \dots; \mathbf{E}\,x_{p_N}] + \mathbf{E}_{pos}
$$
后接 $L$ 层 Transformer block。

## 核心要点
1. 完全去卷积；规模化后大幅超越 ConvNet。
2. 常见尺寸：tiny / small / base / large / huge；patch=14 或 16。
3. 自监督代表：MAE、DINO、I-JEPA 等。

## 代表工作
- ViT (Dosovitskiy 2021)
- [[LeWM]]：用 ViT-tiny (patch 14, 12 层, 192 hidden) 做 encoder
- DINO / DINOv2 / I-JEPA

## 相关概念
- [[Patch Embedding]]
- [[JEPA]]
