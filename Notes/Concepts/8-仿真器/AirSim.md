---
type: concept
aliases: [Microsoft AirSim, Aerial Simulator]
---

# AirSim

## 定义
Microsoft 开源的无人机与地面车辆仿真平台，基于 Unreal Engine 构建，提供逼真的视觉感知和物理仿真，用于 UAV/自动驾驶的感知与控制算法开发。

## 核心要点
1. 基于 Unreal Engine，支持照片级真实感渲染
2. 提供 ROS 接口，支持多模态传感器（RGB、深度、IMU、LiDAR）
3. 支持 sim-to-real transfer，常用于 UAV 导航和自主飞行训练
4. PX4 兼容，支持 SITL（Software-In-The-Loop）仿真

## 代表工作
- [[OpenFlyV2]]: 用 AirSim 训练机载 VLA 模型，在真实 NVIDIA NX 上部署推理

## 相关概念
- [[IsaacLab]]
- [[MuJoCo]]
