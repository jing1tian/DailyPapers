---
type: concept
aliases: [BPS, Basis Point Set, 基点集编码]
---

# BPS（Basis Point Set）

## 定义

一种紧凑的 3D 形状编码方式，将任意物体形状表示为固定数量基点集到物体表面的距离向量，实现形状感知的稠密低维表示。

## 数学形式

给定基点集 $\mathcal{B} = \{b_1, \ldots, b_N\}$ 和物体表面点云 $\mathcal{P}$：

$$
\text{BPS}(\mathcal{P}) = \left\{ \min_{p \in \mathcal{P}} \|b_i - p\|_2 \right\}_{i=1}^N
$$

## 核心要点

1. **固定维度**: 无论物体形状复杂度，输出维度恒为 $N$（通常 512 或 1024），便于网络处理
2. **形状感知**: 相比 bounding box，BPS 能捕捉细粒度几何信息，有助于灵巧抓握
3. **计算效率**: 相比体素或隐式场，BPS 计算轻量，适合实时策略输入
4. **旋转不变性**: 需结合物体坐标系变换使用，本身不具备旋转不变性

## 代表工作

- [[GRAIL]]: 将物体形状以 BPS 编码作为物体感知适配器 $\pi_\phi$ 的输入，使策略感知物体几何信息
- UniDexGrasp: 使用 BPS 编码物体形状用于灵巧抓握策略学习

## 相关概念

- [[人形机器人]]
- [[FoundationPose]]
