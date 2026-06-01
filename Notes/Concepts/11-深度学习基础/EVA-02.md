---
type: concept
aliases: [EVA02, EVA-CLIP]
---

# EVA-02

## 定义
BAAI 提出的大规模视觉编码器，以重建 CLIP 特征为预训练目标（EVA = Explore the Limits of Masked Visual Pre-training），在高分辨率视觉理解任务上达到 SOTA；EVA-02 是使用 EVA-CLIP 特征进行 MIM 的改进版本。

## 数学形式

预训练目标：重建 EVA-CLIP 的视觉特征（替代像素重建）：
$$\mathcal{L} = \sum_{i \in \mathcal{M}} \|f_\theta(\tilde{x})_i - z_i^{\text{CLIP}}\|_2^2$$

其中 $\mathcal{M}$ 为 masked patches，$z_i^{\text{CLIP}}$ 为 CLIP 的 patch embedding。

## 核心要点
1. 预训练目标是重建 CLIP 语义特征而非原始像素，学到更丰富的语义信息
2. 支持高分辨率输入（448×448+），在 OCR、细粒度识别等任务上优于 CLIP
3. 在 Eagle、InternVL 等 MLLM 中被用作高分辨率视觉编码器

## 代表工作
- [[Eagle]]: 混合 EVA-02 + CLIP 的 multi-encoder 设计
- [[InternVL]]: 基于 EVA-02 构建的大规模视觉模型

## 相关概念
- [[CLIP]]
- [[ConvNeXt]]
- [[SAM]]
