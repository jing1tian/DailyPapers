---
type: concept
aliases: [ClearGrasp]
---

# ClearGrasp

## 定义
专为透明物体设计的深度估计/补全方法，利用表面法线、边缘检测和遮挡边界重建透明物体的几何形状，用于机器人抓取。

## 核心要点
1. 透明物体对 RGB-D 传感器无效，ClearGrasp 用 RGB 辅助推断深度
2. 联合预测：表面法线 + 遮挡边界 + 深度补全
3. TransGraspNet 在其基础上做含液体透明器皿的抓取

## 代表工作
- [[TransGraspNet]]: 改进 ClearGrasp 用于含液体透明实验器皿

## 相关概念
- [[深度估计]]
- [[透明物体感知]]
