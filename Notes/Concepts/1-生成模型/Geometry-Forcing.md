---
type: concept
aliases: [Geometry Forcing, 几何强制]
---

# Geometry-Forcing

## 定义

Geometry-Forcing 是一种将几何先验知识通过特征对齐注入视频生成模型的方法，属于表示对齐（Representation Alignment）范式，通过迫使视频骨干的中间特征与几何表示对齐来改善视频的物理一致性。

## 核心要点

1. 无需修改视频模型的输出空间，通过内部特征对齐引入几何约束
2. 属于 REPA（Representation Alignment for Generation）范式的几何应用
3. 与 GEM-4D 同属表示对齐类方法，GEM-4D 进一步引入 4D 动态几何蒸馏
4. 在动态场景中相比 GEM-4D 几何一致性较弱，消融实验显示性能略低于 PAGE-4D 特征监督

## 相关概念

- [[视频生成世界模型]]
- [[Flow Matching]]
- [[几何基础模型]]
- [[GEM-4D]]
