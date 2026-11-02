---
type: concept
aliases: [Multimodal Diffusion Transformer, MM-DiT]
---

# MMDiT

## 定义

MMDiT（Multimodal Diffusion Transformer）是一种用于文本条件图像/视频生成的双流 Transformer 架构，文本 token 和视觉 token 各有独立的权重流，通过交叉注意力或并行注意力融合信息。

## 核心要点

1. **双流设计**: 文本流和图像/视频流使用独立的 MLP 和归一化层，避免直接融合带来的信息干扰
2. **联合注意力**: 两个流的 token 在注意力计算时拼接，共享 QK 计算（但值矩阵分开），实现高效的模态交互
3. **可扩展性**: 双流结构相比单流 DiT 在文本对齐上更优，是 Stable Diffusion 3、Flux 等模型的核心

## 数学形式

设文本 token $x_t$ 和视觉 token $x_v$，联合注意力：

$$
[x_t', x_v'] = \text{Attention}\!\left([x_t; x_v],\; [x_t; x_v]\right)
$$

各流独立经过 MLP 后输出。

## 代表工作

- [[HyNomas]]: 使用 54 个 dual-stream MMDiT blocks 作为视频世界模型 backbone
- Stable Diffusion 3: MMDiT 的提出来源
- Flux: 改进版双流 DiT

## 相关概念

- [[Diffusion Model]]
- [[Attention Mechanism]]
- [[HunyuanVideo]]
