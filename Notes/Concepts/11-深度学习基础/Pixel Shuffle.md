---
type: concept
aliases: [像素重排, Sub-pixel Convolution, 亚像素卷积]
---

# Pixel Shuffle

## 定义

Pixel Shuffle（像素重排）是一种将高通道低分辨率特征图重排为低通道高分辨率特征图（或反之）的操作，常用于视觉模型中压缩空间 Token 数量，同时保留信息密度。

## 数学形式

设输入特征图 $F \in \mathbb{R}^{C \cdot r^2 \times H \times W}$，Pixel Shuffle 输出为：

$$
\text{PixelShuffle}(F) \in \mathbb{R}^{C \times (H \cdot r) \times (W \cdot r)}
$$

反向操作（用于 Token 压缩）：

$$
\text{PixelUnshuffle}(F) \in \mathbb{R}^{C \cdot r^2 \times (H/r) \times (W/r)}
$$

其中 $r$ 为缩放因子。

## 核心要点

1. **无参数操作**: 纯重排，不引入额外参数，计算高效。
2. **信息无损**: 相比池化/插值，不丢失像素信息，只是重新组织排列。
3. **VLM 中的应用**: 在多模态模型中用 PixelUnshuffle 将每帧图像从大量视觉 Token 压缩到少量 Token（如 SmolVLA 将每帧压缩至 64 个 Token），大幅降低 Transformer 序列长度。
4. **超分辨率起源**: 最初由 ESPCN 提出用于图像超分辨率，后被广泛用于 Token 压缩。

## 代表工作

- [[SmolVLA]]: 将每帧视觉 Token 用 PixelShuffle 压缩至 64 个，实现高效 VLA 推理
- [[SmolVLM2]]: 使用 Pixel Shuffle 进行视觉 Token 压缩的 VLM 骨干

## 相关概念

- [[SigLIP]]: 常与 Pixel Shuffle 配合使用的视觉编码器
- [[Vision-Language-Action Model]]: 受益于 Token 压缩的下游应用
- [[Transformer]]: 序列长度决定计算复杂度，Pixel Shuffle 减少序列长度
