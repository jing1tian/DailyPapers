---
type: concept
aliases: [DUSt3R, Dust3R, 密集无约束立体3D重建]
---

# DUSt3R

## 定义

DUSt3R（Dense Unconstrained Stereo 3D Reconstruction）是一种无需已知相机参数的 3D 场景重建基础模型，通过 Transformer 从任意图像对直接回归点图（pointmap），实现端到端的密集 3D 重建。

## 数学形式

给定图像对 $(I^1, I^2)$，DUSt3R 输出两个点图：

$$
\hat{X}^{1,1}, \hat{X}^{2,1} = \text{DUSt3R}(I^1, I^2)
$$

其中 $\hat{X}^{v,1} \in \mathbb{R}^{H \times W \times 3}$ 表示在视角 1 坐标系下的 3D 点坐标。

## 核心要点

1. **无需相机标定**：直接从图像对预测 3D 几何，消除传统立体匹配对相机内外参的依赖
2. **点图表示**：每个像素对应一个 3D 点，密集输出避免稀疏点云的信息损失
3. **全局对齐**：多视角场景通过全局优化将多个点图对齐到统一坐标系
4. **作为几何对齐目标**：其特征可用于蒸馏 3D 几何知识到其他模型（如 [[LAWM-3D]]）

## 代表工作

- [[LAWM-3D]]: 将 DUSt3R 作为对比基线的 3D 对齐目标（LAWM-3D 使用 [[VGGT]] 效果更好 +0.25 PSNR）

## 相关概念

- [[VGGT]]
- [[6-3D视觉]]
- [[FoundationPose]]
