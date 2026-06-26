---
type: concept
aliases: [Intel RealSense, RealSense D435, RealSense 相机]
---

# Intel RealSense

## 定义

Intel RealSense 是 Intel 推出的深度相机系列，其中 D435 型号是机器人操作研究中最常用的 RGB-D 相机之一，提供同步的彩色图像和深度信息，常用于腕部视角或第三方视角的视觉观测采集。

## 核心要点

1. **RGB-D 输出**：同时提供彩色图像和逐像素深度信息
2. **常见部署方式**：腕部安装（wrist camera，跟随末端执行器运动）+ 侧方/第三方视角固定安装，组合提供互补的视觉信息
3. **机器人学习中的角色**：作为 VLA / 操作策略的视觉输入来源，常配合 [[ResNet-18]] 等视觉编码器提取特征

## 代表工作

- [[FORCE]]: 真实世界实验中使用双 RealSense D435（腕部 + 侧视角）采集观测，配合 [[Franka Emika Panda]] 机械臂

## 相关概念

- [[Franka Emika Panda]]
- [[VLA]]
