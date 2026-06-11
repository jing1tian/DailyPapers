---
type: concept
aliases: [Gaussian Splatting, GS, 高斯溅射, 高斯泼溅]
---

# Gaussian Splatting

## 定义

Gaussian Splatting 是一种基于显式三维高斯基元的场景表示与实时可微渲染方法，通过将场景表示为一组各向异性 3D 高斯并用光栅化投影到图像平面，实现高质量新视角合成。

## 数学形式

$$
\hat{V}_{GS} = \mathcal{R}(\mathcal{G}, \{E_i, K_i\}_{i=1}^N)
$$

其中 $\mathcal{G}$ 为高斯场，$E_i, K_i$ 为相机外参和内参，$\mathcal{R}$ 为渲染函数。

## 核心要点

1. **显式表示**: 场景用一组 3D 高斯椭球表示，每个高斯具有位置、协方差、颜色（球谐函数）和不透明度
2. **可微渲染**: 通过高斯投影（splatting）到图像平面，支持梯度回传和端到端优化
3. **实时渲染**: 相比 NeRF，训练和推理速度大幅提升（实时 FPS）
4. **几何评估应用**: 在世界模型评估中用于重建场景三维结构，评估 3D 一致性

## 代表工作

- 3DGS (Kerbl et al., SIGGRAPH 2023): 原始工作
- [[WorldOlympiad]]: 用于几何轨道的三维一致性评估流水线

## 相关概念

- [[3D Gaussian Splatting]]
- [[NeRF]]
- [[Depth Anything 3]]
- [[相机轨迹]]
