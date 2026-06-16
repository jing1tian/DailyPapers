---
type: concept
aliases: [SE(3), Special Euclidean Group, 特殊欧式群, 刚体变换群]
---

# SE(3)（特殊欧式群）

## 定义

SE(3) 是三维空间中所有刚体运动（旋转 + 平移）构成的李群，元素形如 $\mathbf{T} = (\mathbf{R}, \mathbf{t})$，其中 $\mathbf{R} \in SO(3)$ 为旋转矩阵，$\mathbf{t} \in \mathbb{R}^3$ 为平移向量。

## 数学形式

SE(3) 元素可用齐次矩阵表示：

$$
\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}, \quad \mathbf{R} \in SO(3), \; \mathbf{t} \in \mathbb{R}^3
$$

点变换：$\mathbf{p}' = \mathbf{R}\mathbf{p} + \mathbf{t}$

## 核心要点

1. **刚体不变性**: SE(3) 变换保持点间距离和手性（rigid body motion）
2. **李群结构**: 支持组合（矩阵乘法）和求逆操作
3. **6 自由度**: 3 个旋转（roll/pitch/yaw）+ 3 个平移

## 在 3D 视觉/机器人中的应用

- **位姿估计**: 相机位姿、机器人末端位姿均用 SE(3) 表示
- **点云配准**: ICP 等方法求解 SE(3) 对齐变换
- **TraceExtract**: μ₀ 中用 SE(3) 变换进行全局–局部 3D 轨迹对齐，将局部块内估计的轨迹对齐到全局参考系

## 代表工作

- [[mu0]]: 使用 SE(3) 优化对齐相机位姿，消除视频中的相机运动影响
- [[FoundationPose]]: 6-DoF 物体位姿估计（SE(3) 输出）

## 相关概念

- [[Optical Flow]]: 2D 运动估计，SE(3) 是其 3D 对应
- [[MASt3R]]: 基于 SE(3) 变换的 3D 重建方法
