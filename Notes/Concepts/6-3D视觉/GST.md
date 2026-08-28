---
type: concept
aliases: [Gaussian Spatial Tokenizer, 高斯空间分词器]
---

# GST (Gaussian Spatial Tokenizer)

## 定义
将语义特征和深度特征提升为紧凑 3D Gaussian token 的模块，用 learned queries 池化几何显著区域，作为 VLA 的几何感知输入表示。

## 数学形式
$$\mathbf{G} = \text{GST}(f_\text{sem}, f_\text{depth}) = \text{Pool}(\text{3DGaussian}(f_\text{sem} \oplus f_\text{depth}), Q)$$

## 核心要点
1. 将 per-pixel 深度（标量）提升为带方向和置信度的 3D Gaussian token
2. 使用 learned queries 对几何显著区域进行注意力池化
3. 比直接拼接 depth 特征保留更多几何结构信息（表面法向、置信度）

## 代表工作
- [[GaussVLA]]: 首次提出 GST，用于 Geometry-Aware VLA

## 相关概念
- [[VGGT]]
- [[GaussVLA]]
- [[DA-CoT]]
