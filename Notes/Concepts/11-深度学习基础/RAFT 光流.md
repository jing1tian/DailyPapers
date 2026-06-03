---
type: concept
aliases: [RAFT, Recurrent All-Pairs Field Transforms, 光流估计, Optical Flow]
---

# RAFT 光流

## 定义

RAFT（Recurrent All-Pairs Field Transforms）是一种基于循环网络的密集光流估计方法，通过构建全对相关体（all-pairs correlation volume）并利用 GRU 迭代更新光流场，实现高精度的逐像素运动估计。

## 数学形式

在帧 $I_1$ 和 $I_2$ 之间，RAFT 估计每个像素 $p$ 的位移向量 $\mathbf{f} \in \mathbb{R}^{H \times W \times 2}$，使得：

$$
I_1(p) \approx I_2(p + \mathbf{f}(p))
$$

全对相关体（4D 相关体）：
$$
C(p, q) = \langle f_\theta(I_1)[p],\; f_\theta(I_2)[q] \rangle, \quad \forall p, q \in \{1,\ldots,H\} \times \{1,\ldots,W\}
$$

GRU 迭代更新：
$$
h^{k+1} = \mathrm{GRU}(h^k, z^k), \quad \Delta f^k = \mathrm{head}(h^{k+1}), \quad f^{k+1} = f^k + \Delta f^k
$$

## 核心要点

1. **全对相关体（All-Pairs Correlation）**: 计算两帧特征图上所有像素对的点积，形成 $H \times W \times H \times W$ 的相关体，在多尺度金字塔中进行查询
2. **循环更新（Recurrent Update）**: GRU 网络迭代精化光流估计，每步从相关体中查询邻域信息
3. **高精度**: 在 KITTI 等基准上优于传统方法（Lucas-Kanade、Farnebäck），端点误差（EPE）显著更低
4. **实时性**: 在 GPU 上约 20ms/帧（AHEAD 中用于两帧对的光流估计），满足机器人实时控制需求

## 在机器人操作中的应用

AHEAD 中，RAFT 用于估计连续帧间的逐像素光流场 $F_{t-1:t}$，通过 AvgPool 降采样到 patch 级速度 $V_i \in \mathbb{R}^2$，为世界模型提供运动先验：

$$
V_i = \mathrm{AvgPool}_i(F_{t-1:t}), \quad A_i = \frac{V_i - V_i^{\text{prev}}}{\Delta t}
$$

## 代表工作

- [[AHEAD]]: 用 RAFT 光流提供逐 patch 运动估计，驱动潜空间世界模型预测
- [[CoTracker3]]: 基于点轨迹的运动追踪替代方案（AHEAD 消融实验中对比方法）

## 相关概念

- [[场景流]]: 3D 空间中的运动估计（RAFT 的 2D 扩展方向）
- [[CoTracker3]]: 基于 token 的稀疏点轨迹追踪
- [[交叉注意力]]: 现代光流估计中也会使用注意力机制
