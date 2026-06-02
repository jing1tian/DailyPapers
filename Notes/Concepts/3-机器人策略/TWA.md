---
type: concept
aliases: [trajectory-wide attention, TWA]
---

# Trajectory-Wide Attention

## 定义
VLA 运行时监控方法，通过分析整条轨迹的注意力特征而非单帧来检测执行失败。

## 核心要点
1. 1. 轨迹级别特征聚合，而非单帧
2. 2. 结合 OOD 检测（EigenScore, MMD）识别失败模式
3. 3. 免额外训练，接在已有 VLA 上使用

## 代表工作
- [[Hide-and-Seek in Trajectories]]

## 相关概念
- [[OpenVLA]]
- [[LIBERO]]
