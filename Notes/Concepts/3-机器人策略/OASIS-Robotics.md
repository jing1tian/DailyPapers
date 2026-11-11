---
type: concept
aliases: [OASIS, Observation-Action Space Alignment, SE3 Trajectory VLA]
---

# OASIS（机器人版）

## 定义
通过 SE(3) 末端执行器轨迹预测对齐 VLA 的观测空间和动作空间，以 6D 位姿序列作为中间表示改善空间泛化能力。

## 核心要点
1. 插入 SE(3) 轨迹预测层：先预测末端执行器 6D 位姿序列
2. 再通过 IK 将位姿序列转换为关节角度动作
3. SE(3) 表示在不同相机视角和物体位置下更稳定
4. 在 [[LIBERO]] 和 [[CALVIN]] 上验证，含 OOD 场景测试

## 注意
与 Oasis（Minecraft 交互式世界模型）同名但是不同论文。本文为机器人操作方向（2605.25829）。

## 代表工作
- [[OASIS]]: 原始论文（2605.25829 v2）

## 相关概念
- [[VLA]]
- [[LIBERO]]
- [[CALVIN]]
- [[SpatialVLA]]
