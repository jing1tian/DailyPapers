---
type: concept
aliases: [Temporal Gaussian Evolution, 时序高斯演化]
---

# TGE (Temporal Gaussian Evolution)

## 定义
在 3D Gaussian 世界模型中建模操作过程中 Gaussian 参数随时间变化的动态演化模块。

## 数学形式
$$\mathcal{G}_{t+1} = \text{TGE}(\mathcal{G}_t, a_t)$$

## 核心要点
1. 将动作 $a_t$ 作为条件，预测 Gaussian 位置/旋转/透明度的时序变化
2. 支持对物体操作过程中的几何形变进行建模
3. 与 VGGT 等静态 3D 重建模块配合使用

## 代表工作
- [[GaussianDream++]]: 引入 TGE 提升 3D Gaussian WM 对动态操作的建模能力

## 相关概念
- [[GaussianDream]]
- [[VGGT]]
- [[GaussianDream++]]
