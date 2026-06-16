---
type: concept
aliases: [B样条, 三次B样条, Cubic B-Spline, 样条曲线]
---

# B-Spline（B 样条）

## 定义

B 样条（Basis Spline）是由一组称为"控制点"的参数定义的分段多项式曲线，通过基函数的线性组合产生平滑曲线，常用于轨迹表示和运动建模。

## 数学形式

给定 $D$ 个控制点 $\mathbf{P} \in \mathbb{R}^{D \times d}$，B 样条曲线为：

$$
\mathbf{C}(t) = \sum_{i=0}^{D-1} N_{i,k}(t) \cdot \mathbf{P}_i
$$

其中 $N_{i,k}(t)$ 为 $k$ 阶 B 样条基函数，通过递推关系（de Boor 算法）计算。

矩阵形式写作 $\mathbf{C} = \mathbf{B} \mathbf{P}$，$\mathbf{B}$ 为基矩阵。

## 核心要点

1. **局部支撑性**: 每个控制点只影响曲线的局部区间，编辑灵活
2. **平滑性**: 三次（cubic）B 样条具有 $C^2$ 连续性（二阶导数连续）
3. **紧凑表示**: $D=10$ 个控制点可表示长度为数十帧的平滑轨迹，大幅压缩序列长度
4. **平滑正则化**: 通过惩罚二阶差分矩阵 $\mathbf{\Gamma}$ 可控制轨迹平滑度

## 在机器人/轨迹预测中的应用

B 样条常用于轨迹预测任务，将待预测的连续轨迹参数化为少量控制点，再用生成模型（如流匹配、扩散模型）预测这些控制点：

$$
\mathbf{P}_n^\star = \arg\min_{\mathbf{P}} \left\| \mathbf{M}_n \odot \left( \mathbf{B}\mathbf{P} - \mathbf{T}_n \right) \right\|_F^2 + \lambda^2 \left\| \mathbf{\Gamma} \mathbf{P} \right\|_F^2
$$

## 代表工作

- [[mu0]]: 使用三次 B 样条（$D=10$ 控制点）表示 3D 交互轨迹，通过流匹配预测控制点

## 相关概念

- [[Flow Matching]]: 常用于预测 B 样条控制点的生成模型
- [[Diffusion Policy]]: 另一种机器人轨迹生成方法
