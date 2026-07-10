---
type: concept
aliases: [MoveIt2, ROS Motion Planning, 机器人运动规划框架]
---

# MoveIt

## 定义
ROS 生态中最广泛使用的机器人运动规划框架，提供运动学求解、碰撞检测、轨迹规划等功能的统一接口。

## 核心要点
1. 支持多种运动规划算法（OMPL: RRRT*, CHOMP, STOMP 等），通过插件接口切换
2. 内置 KDL/IKFast 正/逆运动学求解器
3. 提供 Python/C++ API，与 RViz 深度集成用于可视化
4. MoveIt2 是 ROS2 版本，目前主流项目正在迁移

## 代表工作
- [[Multi-Agent Robotic Control]]: 在工业巡检多机器人系统中使用 MoveIt 做运动规划

## 相关概念
- [[PDDL]]
- [[MPC]]
