---
type: concept
aliases: [Franka Research 3, FR3, Franka R3, Franka机械臂]
---

# Franka Research 3

## 定义

由 Franka Robotics 推出的 7 自由度（7-DOF）研究级机械臂，是学术机器人操作研究中最广泛使用的平台之一，以高精度力矩传感和开放控制接口著称。

## 核心要点

1. **7 自由度**: 7 个旋转关节，提供灵活的工作空间覆盖和冗余自由度
2. **力矩传感**: 每个关节内置力矩传感器，支持阻抗控制和接触力感知
3. **Franka Control Interface (FCI)**: 提供 1 kHz 实时控制接口，支持关节空间和笛卡尔空间控制
4. **研究定位**: 相比工业机械臂（如 UR5），FR3 专为灵巧操作研究设计，兼顾安全性和可编程性
5. **广泛基准支持**: 是 [[LIBERO]]、[[LIBERO-Plus]] 等众多机器人操作 benchmark 的参考平台

## 代表工作

- [[TS-WAM]]: 在 Franka Research 3 上测试了 Stack Bowls 和 Drawer-Long 任务，IND 场景成功率 75-85%
- [[π0.5]]: 同样在 Franka 平台上进行真实机器人评测
- 众多桌面操作 VLA 论文均使用 FR3 作为评测平台

## 相关概念

- [[LIBERO]]: 基于 Franka 的操作 benchmark
- [[LIBERO-Plus]]: LIBERO 的 OOD 扩展
