---
type: concept
aliases: [Dexterous Teleoperation, 灵巧手遥操作]
---

# DexPilot

## 定义
一种用于灵巧手遥操作的手部重定向系统，将人手的关节角度映射到机器人手的关节配置，实现实时遥控。

## 核心要点
1. 通过视觉感知（RGB 或深度摄像头）估计人手关节位置
2. 基于运动学约束将人手 DoF 映射到机器人手 DoF（跨形态重定向）
3. 最早版本使用优化方法（IK）求解，后续方法（如 SmoothOp）用采样替代

## 代表工作
- [[SmoothOp]]: 提出 SBR 方法替代 DexPilot 的优化方案，实现更低延迟的实时重定向

## 相关概念
- [[MANO]]
- [[GeoRT]]
