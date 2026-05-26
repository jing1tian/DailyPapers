---
type: concept
aliases: [ICP, 迭代最近点算法, Point Cloud Registration]
---

# Iterative Closest Point

## 定义
一种经典点云配准算法，通过迭代求解两组点云之间的最优刚体变换（旋转 + 平移），使对应点之间的距离之和最小。

## 数学形式

$$
T^* = \arg\min_{T \in SE(3)} \sum_{k} \|T(\mathbf{p}_k) - \mathbf{q}_k\|^2
$$

其中 $\mathbf{p}_k$ 为源点云点，$\mathbf{q}_k$ 为目标点云中对应最近点，$T \in SE(3)$ 为刚体变换。

## 核心要点

1. 每次迭代分两步：（1）为源点云各点找最近目标点；（2）求解最优刚体变换（SVD 分解）
2. 收敛到局部最优，对初始位姿敏感，通常需要较好的初始估计
3. 时间复杂度 $O(nk)$（$n$ 为点数，$k$ 为迭代次数），可用 KD-tree 加速
4. 在机器人操作中常用于从观测点云估计物体位姿或末端执行器运动

## 代表工作

- [[GAF]]: 用 ICP 对夹爪区域高斯点集配准，从预测运动场直接估计末端执行器的 SE(3) 变换动作

## 相关概念

- [[3D Gaussian Splatting]]
- [[Gaussian Action Field]]
- [[多视角几何]]
