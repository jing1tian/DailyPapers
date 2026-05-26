---
type: concept
aliases: [MASt3R, Matching And Stereo 3D Reconstruction, MASt3R-SfM]
---

# MASt3R

## 定义
基于 [[Vision Transformer|ViT]] 的多视角 3D 重建基础模型，通过大规模多视角图像对预训练，学习强大的跨视角特征对应和几何关系，可作为下游 3D 任务的视觉骨干。

## 核心要点

1. 基于密集匹配（dense matching）预训练，ViT 编码器能提取具备跨视角几何感知的特征
2. 无需相机标定即可从非结构化图像集合重建 3D 场景（pose-free 重建）
3. 提供强大的多视角特征初始化，可作为 3D 感知任务的预训练权重
4. GAF 使用 MASt3R 的 ViT 权重初始化，加速收敛并提升跨视角特征对齐质量

## 代表工作

- MASt3R 原始论文: 多视角 3D 重建 SOTA 方法
- [[GAF]]: 使用 MASt3R 权重初始化 ViT backbone，训练 80k 步达到高质量 4D 重建

## 相关概念

- [[Vision Transformer]]
- [[多视角几何]]
- [[3D Gaussian Splatting]]
- [[VGGT]]
