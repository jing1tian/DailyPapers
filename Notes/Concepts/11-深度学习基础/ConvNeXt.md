---
type: concept
aliases: [ConvNeXt V1, ConvNeXt V2]
---

# ConvNeXt

## 定义
Meta 提出的纯卷积视觉骨干网络，系统性地借鉴 Vision Transformer 的设计原则（大卷积核、LayerNorm、GELU 等）来改造标准 ResNet，在 ImageNet 分类和下游任务上与 ViT 持平或超越。

## 数学形式

ConvNeXt block（depthwise + pointwise，类似 inverted bottleneck）：
$$y = x + \text{PW}(\text{GELU}(\text{PW}(\text{LN}(\text{DW-Conv}(x)))))$$

其中 DW-Conv 使用 $7\times7$ depthwise 卷积，PW 为 $1\times1$ pointwise 卷积。

## 核心要点
1. 将 ResNet 的 $3\times3$ conv 替换为 $7\times7$ depthwise conv，扩大感受野
2. 采用 LayerNorm 替代 BatchNorm，更适合小 batch size 训练
3. 用 GELU 替代 ReLU，FFN 扩展比例从 4x 降至 4x（与 ViT 一致）
4. ConvNeXt-V2 引入 masked autoencoder 预训练（FCMAE）

## 代表工作
- [[Eagle]]: 使用 ConvNeXt 作为视觉编码器之一，与 CLIP 混合

## 相关概念
- [[CLIP]]
- [[EVA-02]]
