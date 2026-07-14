---
type: concept
aliases: [Pixel Splat]
---

# PixelSplat

## 定义
前馈 3D Gaussian Splatting 方法，从两张输入图像直接预测 3D Gaussian，实现实时新视图合成，无需逐场景优化。

## 数学形式
$$\mathcal{G} = f_\theta(I_1, I_2, P_1, P_2)$$

其中 $f_\theta$ 是编码器网络，直接输出 Gaussian 参数集合。

## 核心要点
1. 双视图输入，前馈出 3D Gaussians
2. 用 epipolar 对应关系引导深度预测
3. 实时渲染，无需 NeRF 式的体积渲染
4. 开创了"可泛化 3DGS"方向

## 代表工作
- [[DepthSplat]]：在 PixelSplat 基础上改进深度估计
- [[StereoSplat+]]：单视角立体版本

## 相关概念
- [[3D Gaussian Splatting]]
- [[DepthSplat]]
