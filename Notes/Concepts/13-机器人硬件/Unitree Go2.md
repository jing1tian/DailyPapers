---
type: concept
aliases: [Go2, Unitree Go2 Edu]
---

# Unitree Go2

## 定义
宇树科技（Unitree Robotics）开发的消费级/研究级四足机器人，搭载高扭矩关节电机和板载计算单元，支持扩展搭载机械臂和外置传感器，广泛用于机器人操作与导航研究。

## 核心要点
1. **Edu 版本**: 支持二次开发，可扩展 6-DoF 机械臂和腕部相机
2. **板载算力**: Jetson Orin NX（用于低层控制），搭配离线 GPU 服务器做感知/规划
3. **运动能力**: 支持站姿/蹲姿切换，适应不同高度操作需求
4. **通信**: ROS/WiFi 与离线计算节点通信，延迟约 0.05 s/chunk

## 代表工作
- [[EmbodiedMG]]: Go2 Edu + 6-DoF 臂 + 腕部 RGB 相机，完成开放词汇移动操作任务

## 相关概念
- [[Loco-Manipulation]]
- [[人形机器人]]
