---
type: concept
aliases: [Chamfer Distance, CD, 倒角距离]
---

# Chamfer Distance

## 定义

衡量两个点云 $S_1$ 和 $S_2$ 之间差异的对称距离度量，对每个点计算到另一点云的最近邻距离之和。

## 数学形式

$$
\text{CD}(S_1, S_2) = \frac{1}{|S_1|}\sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2^2 + \frac{1}{|S_2|}\sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2^2
$$

单向版本（depth-only）只计算从 $S_1$ 到 $S_2$ 的方向：

$$
\text{CD}_{z}(S_1, S_2) = \frac{1}{|S_1|}\sum_{x \in S_1} \min_{y \in S_2} \|x_z - y_z\|^2
$$

## 核心要点

1. **对称性**: 标准 CD 是对称的，考虑双向匹配；单向 CD 仅考虑一个方向，用于接触约束
2. **可微性**: CD 关于点坐标可微，适合作为深度学习训练的损失函数
3. **应用场景**: 点云配准、形状重建、3D 生成模型评估、HOI 重建中的深度对齐

## 代表工作

- [[GRAIL]]: 用 CD 实现深度对齐损失（$\mathcal{L}_{depth}$）和接触损失（$\mathcal{L}_{cont}$），约束 4D HOI 重建
- [[FoundationPose]]: 物体姿态追踪中用于点云配准

## 相关概念

- [[FoundationPose]]
- [[人形机器人]]
