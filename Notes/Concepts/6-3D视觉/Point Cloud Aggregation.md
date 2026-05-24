---
type: concept
aliases: [点云聚合, 全局点云扩展]
---

# Point Cloud Aggregation

## 定义

Point Cloud Aggregation 是将多个视角/帧生成的局部点云合并为统一全局点云表示的过程，用于构建场景级别的 3D 先验。

## 数学形式

$$
\mathbf{P}^{glo} = [\mathbf{P}^{ref}, \hat{\mathbf{P}}] \in \mathbb{R}^{(N + \hat{N}) \times 3}
$$

## 核心要点

1. **多源拼接**: 将参考视图点云与额外新视图点云直接拼接
2. **全局表示**: 聚合后的点云覆盖更大场景范围，可渲染为任意视角的几何先验
3. **记忆机制基础**: 全局点云是 Global-Geometric Memory（GGM）的核心数据结构

## 符号说明

- $\mathbf{P}^{ref}$: 参考视图的点云（$N$ 个点）
- $\hat{\mathbf{P}}$: 额外新视图生成的点云（$\hat{N}$ 个点）
- $\mathbf{P}^{glo}$: 拼接后的全局点云

## 代表工作

- [[HYWorld2]]: WorldStereo 2.0 中构建 GGM 全局几何记忆，用于引导多视图生成保持 360° 结构一致性

## 相关概念

- [[Point Cloud Warping]]
- [[3D Gaussian Splatting]]
- [[NavMesh]]
