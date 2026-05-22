---
type: concept
aliases: [TGE, 时序高斯演化模块, Temporal Gaussian Evolution Module]
---

# Temporal Gaussian Evolution

## 定义

GaussianDream 中提出的时序特征交互模块，通过交替执行帧内自注意力和跨帧时间注意力，将多帧 [[VGGT]] 特征与可学习高斯查询融合，生成包含时序运动线索的紧凑 GaussianDream 前缀。

## 数学形式

$$
Z_t^{GD} = \text{Proj}_{512 \to 2048}\!\left[\text{TGE}\!\left(\text{Proj}_{2048 \to 512}(Q_{GD}),\; \{P_{t-K:t}^{(m)}\}\right)\right]_t
$$

其中 TGE 内部每个注意力块依次执行帧内自注意力、时间注意力和 MLP。

## 核心要点

1. **结构**: 12 个注意力块，8 头，工作维度 512（输入输出通过线性投影与 2048 对齐）
2. **帧内自注意力（frame-wise token interaction）**: 在每一帧内，GaussianDream 查询与该帧的 VGGT patch token 进行自注意力交互，聚合当前帧的空间信息
3. **时间注意力（temporal attention）**: 跨多帧（如 $t-10, t-5, t$）对同一 token 位置做时间维度注意力，建模帧间运动变化
4. **输出取当前帧**: 取 $t$ 帧对应的输出 token 作为 GaussianDream 前缀 $Z_t^{GD}$，隐式编码时序运动先验

## 代表工作

- [[GaussianDream]]: 首次提出 TGE，用于 3D 高斯世界模型与 VLA 策略的联合训练

## 相关概念

- [[VGGT]]
- [[3D Gaussian Splatting]]
- [[Cross-Attention]]
