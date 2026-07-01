---
type: concept
aliases: [Simultaneous Localization and Mapping, 同时定位与建图]
---

# SLAM

## 定义
机器人在未知环境中同时估计自身位姿（定位）并构建环境地图（建图）的技术框架。

## 数学形式
$$p(\mathbf{x}_{1:t}, \mathbf{m} \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$$

其中 $\mathbf{x}_{1:t}$ 为轨迹，$\mathbf{m}$ 为地图，$\mathbf{z}$ 为观测，$\mathbf{u}$ 为控制输入。

## 核心要点
1. 前端：特征提取/匹配，里程计估计
2. 后端：因子图优化或粒子滤波
3. 主要方法：LiDAR SLAM（LOAM）、视觉 SLAM（ORB-SLAM）、3DGS-SLAM

## 代表工作
- [[J-LAW]]: 把 SLAM 和 world model 统一到 latent factor graph
- [[CubifyGS]]: 3DGS-based lifelong SLAM

## 相关概念
- [[MonoGS]]
- [[SplaTAM]]
- [[ICP]]
