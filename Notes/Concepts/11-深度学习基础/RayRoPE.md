---
type: concept
aliases: [Ray Rotary Position Encoding]
---

# RayRoPE

## 定义
RayRoPE：将 camera ray 方向作为 RoPE 位置编码基础的方法，使 transformer 的 attention 在 3D 物理空间中具备几何感知能力，而非只有 2D grid 的位置关系。

## 数学形式
对 query/key 向量按 ray 方向 $\mathbf{r}$ 做旋转变换：

$$q_i' = q_i \cdot e^{i \langle \mathbf{p}_i, \mathbf{r}_i \rangle}$$

其中 $\mathbf{p}_i$ 为 3D 物理位置，$\mathbf{r}_i$ 为对应 camera ray。

## 核心要点
1. 标准 RoPE 用 2D grid index 编码位置，忽略 3D 几何关系
2. RayRoPE 将 camera ray 方向和物理坐标嵌入位置编码
3. 特别适合多视角输入和物理感知的 robot manipulation

## 代表工作
- [[POD-RoPE]]: 进一步扩展为 Physical Observation Distribution RoPE

## 相关概念
- [[RoPE]]
- [[POD-RoPE]]
- [[LVSM]]
