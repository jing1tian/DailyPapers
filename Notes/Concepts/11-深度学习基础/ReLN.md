---
type: concept
aliases: [Rotation-Equivariant LayerNorm]
---

# ReLN

## 定义
Rotation-Equivariant LayerNorm：在旋转等变神经网络中替代标准 LayerNorm 的归一化层，保持 SO(3)/SE(3) 等变性。

## 核心要点
1. 标准 LayerNorm 破坏旋转等变性（对激活统计量归一化会混淆方向信息）
2. ReLN 仅对不变量（范数）归一化，保留方向信息
3. 用于 [[E3DGS]] 等等变 3DGS 架构

## 代表工作
- [[E3DGS]]: Color-as-Geometry embedding + ReLN

## 相关概念
- [[PointNet]]
- [[ManiGaussian]]
