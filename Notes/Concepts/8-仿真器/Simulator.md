---
type: concept
aliases: [仿真环境, Simulation Environment, Sim]
---

# Simulator

## 定义

通过物理引擎模拟机器人与物体交互的虚拟环境，提供可重置、安全的训练和评估平台。

## 核心要点

1. **可重置性**：支持反复尝试同一任务，是 WICL 世界反馈的前提条件
2. **安全排练**：避免在真实硬件上测试危险动作
3. **特权信息**：提供真值物体位姿、接触力等部署时不可得的信息（需严格隔离）
4. **与真实的差距**：sim-to-real gap 需通过域随机化、语义迁移等方法弥合

## 代表工作

- [[RoboTwin 2.0]]: 50 双臂任务的操作仿真基准
- [[RoboScript]]: 将仿真器作为 WICL 中的"世界"

## 相关概念

- [[Sim-to-Real Transfer]]
- [[World In-Context Learning]]
- [[RoboTwin 2.0]]
