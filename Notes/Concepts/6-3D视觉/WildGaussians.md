---
type: concept
---

# WildGaussians

## 定义
处理 in-the-wild 场景（动态物体、外观变化）的 3D Gaussian Splatting 方法，用语义特征解耦静态/动态部分。

## 数学形式
$$\alpha_i^{dyn} = \text{sigmoid}(f_{sem}(x_i) \cdot w_{dyn})$$

## 核心要点
1. MoMaSplat 用它处理操控中运动物体的 3DGS 建模
2. 比标准 3DGS 更鲁棒于遮挡和动态变化

## 代表工作
- 见关联笔记

## 相关概念
- [[FastVGGT]]
- [[MoMaSplat]]
