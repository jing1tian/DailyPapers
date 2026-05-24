---
type: concept
aliases: [深度转法线, 深度法线转换, Depth-to-Normal]
---

# Depth-to-Normal Conversion

## 定义

从预测的深度图通过反投影与叉积运算导出表面法线的技术，用于在无显式法线标签时提供几何一致性监督。

## 数学形式

$$
\tilde{N}_i(x) = \text{normalize}\left(\frac{\partial \mathbf{P}_i}{\partial u} \times \frac{\partial \mathbf{P}_i}{\partial v}\right), \quad \mathbf{P}_i = \mathbf{K}^{-1} \hat{D}_i \cdot [u, v, 1]^\top
$$

其中 $\hat{D}_i$ 为预测深度图，$\mathbf{K}$ 为相机内参矩阵，$u, v$ 为像素坐标。

## 核心要点

1. 将深度图反投影为 3D 点云，再对相邻像素方向求叉积得到法线
2. 使法线监督与深度预测耦合，改善几何一致性
3. 在仅有深度 GT 而无法线 GT 时，可作为额外几何监督信号

## 代表工作

- [[HYWorld2]]: WorldMirror 2.0 使用 Depth-to-Normal Loss 耦合深度和法线监督

## 相关概念

- [[Depth-to-Normal Loss]]
- [[3D Gaussian Splatting]]
- [[Point Cloud Warping]]
