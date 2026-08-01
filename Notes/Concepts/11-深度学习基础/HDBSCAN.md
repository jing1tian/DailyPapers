---
type: concept
aliases: [Hierarchical DBSCAN, 层次密度聚类]
---

# HDBSCAN

## 定义
Hierarchical Density-Based Spatial Clustering of Applications with Noise，一种基于层次密度的聚类算法，DBSCAN 的层次化扩展，可自动确定聚类数量并对噪声点鲁棒。

## 核心要点
1. 与 DBSCAN 不同，无需预设 epsilon 半径，通过构建层次树自动选取最稳定的聚类
2. 通过"聚类稳定性"指标从层次树中提取最优聚类划分
3. 对噪声/异常点鲁棒，将低密度点标记为 noise 而非强制归入某类
4. 在 robotics 中用于失败轨迹聚类（如 [[RedFlow]] 对失败模式分类）

## 相关概念
- [[DBSCAN]]
- [[RedFlow]]
