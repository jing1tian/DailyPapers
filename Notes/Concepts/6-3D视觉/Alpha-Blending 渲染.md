---
type: concept
aliases: [Alpha-Blending, Volume Rendering, 体渲染, 3DGS 渲染, Gaussian Splatting Rendering]
---

# Alpha-Blending 渲染

## 定义
[[3D Gaussian Splatting]] 中的可微渲染机制，将场景中所有高斯原语按深度排序后沿射线进行 alpha 混合，累积得到最终像素颜色，支持梯度反向传播优化高斯参数。

## 数学形式

$$
C(\mathbf{p}) = \sum_{i=1}^{N} \alpha_i c_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

其中 $\alpha_i$ 为第 $i$ 个高斯在 2D 投影上的密度（由协方差矩阵 $\Sigma_i$ 和不透明度 $\sigma_i$ 决定）。

## 核心要点

1. 高斯原语按从近到远排序，前景遮挡后景（$\prod_{j<i}(1-\alpha_j)$ 为透射率）
2. $\alpha_i$ 由高斯的 2D 投影协方差矩阵 $\Sigma_i' = J W \Sigma_i W^T J^T$ 决定，支持可微渲染
3. 相比 NeRF 的射线采样，3DGS 使用光栅化加速渲染，速度快 100 倍以上
4. 可同时渲染当前帧和通过位移 $\Delta\mu$ 更新后的未来帧，支持时序建模

## 代表工作

- [[3D Gaussian Splatting]]: 原始提出
- [[GAF]]: 对当前帧和未来帧（加入运动位移后）分别渲染，计算 LPIPS + MSE 联合损失

## 相关概念

- [[3D Gaussian Splatting]]
- [[Gaussian Action Field]]
- [[NeRF]]
