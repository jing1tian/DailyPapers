---
type: concept
aliases: [SMPL, Skinned Multi-Person Linear Model, SMPL-X]
---

# SMPL (Skinned Multi-Person Linear Model)

## 定义
MPI-IS 提出的参数化人体三维模型，将人体形状分解为姿态参数（θ，关节旋转）和形状参数（β，体型），支持高效的可微分人体重建和生成。

## 数学形式
$$
M(\theta, \beta) = W(T_p(\theta, \beta), J(\beta), \theta, \mathcal{W})
$$
其中 $T_p$ 为 pose-corrective 的 template mesh，$J$ 为关节位置，$\mathcal{W}$ 为 blend weights，$W$ 为 LBS（Linear Blend Skinning）。

## 核心要点
1. 姿态参数 θ ∈ R^{72}（24 关节 × 3 轴角）+ 形状参数 β ∈ R^{10}（PCA 系数）
2. 可微分：支持从视频/图像优化人体参数
3. 扩展：SMPL-X 加入手部和面部参数
4. 广泛用于人体运动重建、数据增广、仿真

## 代表工作
- Loper et al. (2015): "SMPL: A Skinned Multi-Person Linear Model"
- [[EgoExo-WM]]: 用 SMPL 从 exo 视频重建人体运动以生成 ego 训练数据

## 相关概念
- [[V-JEPA]]
- [[EPIC-KITCHENS]]
