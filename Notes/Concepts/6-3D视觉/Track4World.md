---
type: concept
aliases: [World-Centric 3D Object Tracker]
---

# Track4World

## 定义
Track4World：以世界坐标系为中心的 3D 物体追踪模型，对场景中的 scene/motion/camera 三类信息进行联合 token 化和追踪。

## 核心要点
1. 以世界坐标（而非机器人末端或相机坐标）表示物体位置
2. 输出 scene token、motion token、camera token 三类表征
3. 冻结后作为 teacher 网络对 VLA 蒸馏 3D 空间理解

## 代表工作
- [[Track4Action]]: 将 Track4World 的 3D 知识蒸馏进 VLA

## 相关概念
- [[Track4Action]]
- [[SpatialVLA]]
