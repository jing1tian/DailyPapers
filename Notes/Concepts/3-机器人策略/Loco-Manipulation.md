---
type: concept
aliases: [Loco-Manipulation, 移动操作, 运动与操作]
---

# Loco-Manipulation

## 定义

Loco-Manipulation 是将**移动运动（Locomotion）**与**物体操作（Manipulation）**相结合的机器人任务类型，要求机器人在行走过程中同时执行抓取、放置、搬运等操作任务。

## 核心要点

1. **移动中操作**: 机器人需在行走状态下保持末端执行器稳定，完成物体交互
2. **长时程 3D 追踪**: 目标物体需要在行走、接触、遮挡、恢复全过程中保持可寻址状态
3. **全身协调**: 腿部控制与手臂操作需要协同规划（Whole-Body Control）
4. **环境适应性**: 需处理运动引起的视角变化、振动、物体遮挡等干扰

## 主要挑战

- 行走振动导致 RGB-D 深度不稳定
- 物体在移动过程中可能被身体遮挡
- 全身关节数量多，动作空间维度高
- 感知-动作-验证三者难以保持状态一致（见物体状态分叉问题）

## 代表工作

- [[POT-VLA]]: 通过持久化 3D 物体 Token 解决 Loco-Manipulation 中的感知-动作状态分叉
- [[Being-H0.7]]: Being-0 系统，模块化 Loco-Manipulation 方法
- [[人形机器人]]: 人形平台（如 Unitree G1）是主要研究载体
- [[Omega0]]: ω-0，通过潜在预测世界-动作模型实现并发人形运动-操作，11 任务平均 SR 81.8%（无分解式架构）

## 相关概念

- [[VLA]]
- [[人形机器人]]
- [[Action Chunking]]
- [[Predicate Supervisor]]
