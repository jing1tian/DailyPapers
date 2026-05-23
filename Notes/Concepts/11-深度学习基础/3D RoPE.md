---
type: concept
aliases: [3D Rotary Position Embedding, 三维旋转位置编码, 3D RoPE]
---

# 3D RoPE

## 定义

[[Rotary Position Encoding|RoPE]] 的三维扩展，将 token 的空间坐标 $(x, y)$ 和时间步 $t$ 分别编码到 Query/Key 向量中，使注意力分数仅依赖相对空间位移和相对时间差，适用于视频/帧序列建模。

## 数学形式

对 token 的 Query 向量 $\mathbf{q}_i$（位于空间位置 $(x_i, y_i)$、时间步 $t_i$）：

$$
\mathbf{q}'_i = \begin{pmatrix} R_{x(i), y(i)} & \mathbf{0} \\ \mathbf{0} & R_{t(i)} \end{pmatrix} \mathbf{q}_i
$$

其中前半段维度用空间坐标 $(x, y)$ 旋转，后半段用时间坐标 $t$ 旋转。点积结果：

$$
{\mathbf{q}'_i}^\top \mathbf{k}'_j = \mathbf{q}_i^\top \begin{pmatrix} R_{x(j)-x(i),\ y(j)-y(i)} & \mathbf{0} \\ \mathbf{0} & R_{t(j)-t(i)} \end{pmatrix} \mathbf{k}_j
$$

注意力分数仅依赖相对偏移 $(\Delta x, \Delta y, \Delta t)$，实现空间和时间上的平移不变性。

## 核心要点

1. 将嵌入维度分为两部分：前 $D/2$ 维编码空间位置 $(x, y)$，后 $D/2$ 维编码时间步 $t$
2. 空间旋转矩阵 $R_{x,y}$ 使模型感知 token 在帧内的相对位置，有助于光流式 token 对应
3. 时间旋转矩阵 $R_t$ 使模型感知帧间的时序关系
4. 与标准 1D RoPE 完全兼容，可替换而无需修改架构

## 代表工作

- [[ITC]]: 在 Token-based World Model 中引入 3D RoPE，提升跨帧 token 对应精度

## 相关概念

- [[Rotary Position Encoding]]: 标准 1D RoPE，3D RoPE 的基础
- [[World Model]]: 3D RoPE 的主要应用场景
- [[Block Causal Attention]]: 常与 3D RoPE 配合使用于帧序列建模
