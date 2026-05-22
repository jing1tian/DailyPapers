---
type: concept
aliases: [CARLA Simulator, 自动驾驶仿真器]
---

# CARLA

## 定义
开源自动驾驶仿真平台，基于 Unreal Engine 构建，支持多传感器模拟（RGB、LiDAR、雷达、语义分割）、天气模拟、多智能体交互，是自动驾驶算法评估的标准平台。

## 核心要点
1. 开源免费，API 完善，支持 Python
2. 支持传感器降级模拟（雾、雨、遮挡），适合安全性研究
3. 支持闭环评估（CARLA Leaderboard）
4. 多 agent 场景：行人、车辆、自行车
5. 与 NAVSIM 等 benchmark 配合使用

## 代表工作
- [[Lost-in-Fog]]：在 CARLA 中注入传感器扰动，评估驾驶 VLA 的推理脆性

## 相关概念
- [[NAVSIM]]
- [[DriveLM]]
