---
type: concept
aliases: [深度法线一致性损失, d2n loss]
---

# Depth-to-Normal Loss

## 定义

Depth-to-Normal Loss（$\mathcal{L}_{d2n}$）是一种几何一致性损失，通过将预测深度图反投影为 3D 点云后计算法线，再与目标法线计算角度误差，显式耦合深度预测与表面法线预测的几何一致性。

## 数学形式

首先从深度派生法线：

$$
\tilde{N}_i(x) = \text{normalize}\left(\frac{\partial \mathbf{P}_i}{\partial u} \times \frac{\partial \mathbf{P}_i}{\partial v}\right), \quad \mathbf{P}_i = \mathbf{K}^{-1} \hat{D}_i \cdot [u, v, 1]^\top
$$

然后计算角度误差损失：

$$
\mathcal{L}_{d2n} = \frac{1}{|\mathcal{V}|} \sum_{x \in \mathcal{V}} \arccos\left(\frac{\tilde{N}_i(x) \cdot \hat{N}_i(x)}{\|\tilde{N}_i(x)\| \|\hat{N}_i(x)\|}\right)
$$

## 核心要点

1. **几何一致性**: 强制深度预测与法线预测在几何上相互一致，避免两者各自优化导致的矛盾
2. **跨域监督**: 在合成数据上使用 GT 法线，在真实数据上使用伪法线（来自教师模型）
3. **有效像素掩码**: 仅对有效像素集合 $\mathcal{V}$ 计算损失，排除无效区域（如透明物体、传感器噪声）

## 代表工作

- [[HYWorld2]]: WorldMirror 2.0 引入 Depth-to-Normal Loss 作为显式几何约束，显著提升深度几何一致性

## 相关概念

- [[3D Gaussian Splatting]]
- [[Point Cloud Warping]]
