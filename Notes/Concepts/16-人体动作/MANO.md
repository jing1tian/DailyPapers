---
type: concept
aliases: [MANO Hand Model, 手部参数化模型]
---

# MANO

## 定义

MANO（**M**odel with **A**rticulated **N**eural **O**bjects for hand）是一种可微参数化人手 3D 形状与姿态模型，将手部表示为低维形状和姿态参数的函数，广泛用于手部动作估计与第一视角视频理解。

## 数学形式

MANO 将手部顶点表示为：
$$\mathbf{V} = W(T(\beta, \theta), J(\beta), \theta, \mathcal{W})$$

其中 $\beta$ 为形状参数，$\theta$ 为姿态参数（关节角），$\mathcal{W}$ 为蒙皮权重。

## 核心要点

1. 提供 21 个关节点（腕部 + 5 指 × 4 关节）的运动学拓扑，与机械臂 URDF 格式兼容
2. 可直接用于 [[Forward Kinematics]] 投影，与机器人骨架渲染使用完全相同的流程
3. 被广泛用于第一视角手部动作理解（EgoDex, EPIC-Kitchens 等数据集的标注）

## 代表工作

- [[OSCAR]]: 使用 MANO 拓扑将人手骨架与机械臂骨架统一，实现跨本体条件编码

## 相关概念

- [[Forward Kinematics]]
- [[EgoDex]]
- [[SMPL]]
