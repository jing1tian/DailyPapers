---
type: concept
aliases: [点云投影, Point Cloud Projection]
---

# Point Cloud Warping

## 定义

Point Cloud Warping 是将图像像素通过相机参数和深度图反投影到 3D 世界坐标系，生成点云表示的过程，常用于构建多视图几何先验或全局 3D 记忆。

## 数学形式

$$
\mathbf{P}_i^{tar}(x) \simeq \mathbf{R}_i^{c \to w} D(x) \mathbf{K}_i^{-1} \hat{x}
$$

## 核心要点

1. **反投影**: 利用相机内参 $\mathbf{K}^{-1}$ 和深度 $D(x)$ 将 2D 像素映射到 3D 点
2. **坐标变换**: 通过旋转矩阵 $\mathbf{R}^{c \to w}$ 转换到世界坐标系
3. **多视图聚合**: 多帧点云可拼接为全局点云用于场景表示

## 符号说明

- $\mathbf{R}_i^{c \to w}$: 相机到世界坐标系的旋转矩阵
- $D(x)$: 像素 $x$ 处的深度值（单目或 GT）
- $\mathbf{K}_i$: 相机内参矩阵
- $\hat{x}$: 齐次像素坐标 $[u, v, 1]^\top$

## 代表工作

- [[HYWorld2]]: WorldStereo 2.0 中用于构建 GGM（全局几何记忆），WorldMirror 2.0 中用于 Depth-to-Normal 转换

## 相关概念

- [[Point Cloud Aggregation]]
- [[3D Gaussian Splatting]]
- [[Depth-to-Normal Loss]]
