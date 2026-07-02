---
type: concept
aliases: [Sub-JEPA, Subspace JEPA, 子空间JEPA]
---

# Sub-JEPA

## 定义

[[JEPA]] 的子空间正则化变体，通过约束潜变量分布在多个子空间中满足高斯分布（而非整体高斯），防止 [[Representation Collapse|表示坍缩]]，同时保留比 VICReg 更细粒度的表示约束。

## 核心要点

1. 将潜空间分成多个子空间，分别施加高斯正则化，比整体协方差正则化更灵活
2. 属于无重建 JEPA 类世界模型，避免像素级解码开销
3. 在 Two-Room 和 Reacher 上表现较强，但在 Push-T 上明显弱于其他方法（规划成功率仅 63.73%）

## 代表工作

- [[Delta-JEPA]]: 在规划成功率上全面超越 Sub-JEPA，提出以位移监督替代子空间正则化

## 相关概念

- [[JEPA]]
- [[LeWM]]
- [[PLDM]]
- [[Representation Collapse]]
- [[VICReg]]
