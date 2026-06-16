---
type: concept
aliases: [AstriBot S1, AstriBot]
---

# AstriBot S1

## 定义

一种双臂机器人平台，由 Beijing Innovation Center of Humanoid Robotics 开发，用于接触密集型操作任务的真实世界评估。

## 核心要点

1. **双臂配置**：支持双臂协作操作，动作空间为 16D（每臂：3D 位置 + 四元数 + 夹爪值）
2. **接触密集任务评估**：适用于抓取（Plate/Bottle）、精密插入（Pen）、拼搭（Lego）等任务
3. **标准评估规模**：典型设置为每任务 100 次演示数据，10 次 rollout 评估

## 代表工作

- [[WAM4D]]: 使用 AstriBot S1 进行 4 个任务的真实世界 WAM 策略评估

## 相关概念

- [[LingBot-VA]]
- [[World Action Model]]
