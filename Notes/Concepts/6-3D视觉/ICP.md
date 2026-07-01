---
type: concept
aliases: [Iterative Closest Point, 迭代最近点]
---

# ICP（Iterative Closest Point）

## 定义
通过迭代配准两组点云来估计 6-DOF 相对位姿的经典算法。

## 数学形式
$$R^*, t^* = rg\min_{R,t} \sum_i \|R p_i + t - q_{c(i)}\|^2$$

## 核心要点
1. 交替执行：最近点匹配 → 最小二乘位姿求解
2. 对初始化敏感，在遮挡和低重叠度下容易 drift
3. 常作为精配准步骤，粗配准后再用 ICP

## 代表工作
- [[CubifyGS]]: 用 ICP 做 object pose 更新

## 相关概念
- [[SLAM]]
