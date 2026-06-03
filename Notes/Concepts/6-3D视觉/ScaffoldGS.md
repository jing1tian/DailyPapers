---
type: concept
aliases: [ScaffoldGS, Scaffold-GS]
---

# ScaffoldGS

## 定义
基于锚点结构的 3D Gaussian Splatting 变体，用稀疏锚点生成局部 Gaussian 分布，提升重建质量和内存效率。

## 核心要点
1. 每个锚点预测周围的 Gaussian 属性（位置、颜色、不透明度）
2. 比标准 3DGS 减少冗余 Gaussian 数量
3. MRO-GWM 基于 ScaffoldGS 构建 object-centric 世界模型

## 代表工作
- [[MRO-GWM]]: 用 ScaffoldGS 作为刚体物体的表示骨干

## 相关概念
- [[3D Gaussian Splatting]]
- [[MRO-GWM]]
