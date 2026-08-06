---
type: concept
aliases: [World-Centric 3D Tracker]
---

# Track4World

## 定义
Track4World：一个以世界为中心（而非以机器人末端执行器为中心）的 3D 物体追踪器，用于追踪场景中所有物体的运动状态。

## 核心要点
1. 编码 scene/motion/camera token 三类空间信息
2. 以世界坐标系而非相机坐标系进行追踪
3. 冻结后作为 teacher 对 VLA 进行知识蒸馏

## 代表工作
- [[Track4Action]]: 将 Track4World 的 3D 空间理解能力蒸馏进 VLA

## 相关概念
- [[Track4Action]]
- [[SpatialVLA]]
