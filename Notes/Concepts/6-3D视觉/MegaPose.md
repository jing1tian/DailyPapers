---
type: concept
aliases: [MegaPose]
---

# MegaPose

## 定义
Meta 提出的大规模 6DoF 物体姿态估计框架，通过在数百万合成渲染图像上预训练，实现对新颖物体的 zero-shot 姿态估计（仅需 CAD 模型或参考图像）。

## 核心要点
1. 训练数据：数百万合成渲染（ShapeNet + 真实背景合成）
2. 两阶段：粗估计（分类候选姿态）+ 精化（render-and-compare 迭代）
3. Zero-shot 泛化：训练时未见的物体直接用 3D 模型/参考图推理
4. 对遮挡和光照变化相对鲁棒
5. FoundationPose 是其后续改进工作

## 代表工作
- [[ComPose]]: 与 MegaPose 对比，测试手遮挡下的姿态追踪

## 相关概念
- [[FoundationPose]]
- [[DexYCB]]
