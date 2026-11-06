---
type: concept
aliases: [Archive of Motion Capture as Surface Shapes]
---

# AMASS

## 定义
大规模人体动作捕捉数据集，统一了多个动捕数据库的骨架表示，用 [[SMPL]] 参数化表示人体姿态。

## 核心要点
1. 合并了 15+ 个动捕数据集，超过 300 小时的人体运动
2. 统一用 SMPL 参数（shape + pose）表示，消除各数据库骨架差异
3. 是人体动作生成、运动重定向、物理仿真 baseline 的常用数据来源

## 代表工作
- [[GRASP]]: 用 AMASS 做人类动作到机器人的 physics-based retargeting

## 相关概念
- [[SMPL]]
- [[OakInk2]]
- [[GMR]]
