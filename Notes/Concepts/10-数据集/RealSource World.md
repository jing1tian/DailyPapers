---
type: concept
aliases: [RealSource World Dataset]
---

# RealSource World

## 定义

由 Midea Group AIRC 和 Tongji University 构建的大规模真实双臂操作视频数据集，用于机器人 World Model 的预训练。

## 核心要点

1. **规模**：11,428 条轨迹，1,400 万+ 帧，覆盖 35 个操作任务
2. **视角**：多视角采集（头部相机 + 双腕相机）
3. **平台**：双臂机器人操作台面任务场景
4. **用途**：为 PiL-World 提供预训练数据，使模型习得通用机器人-环境动力学

## 代表工作

- [[PiL-World]]: 使用 RealSource World 进行 22 epoch 预训练（64 块 H20 GPU）

## 相关概念

- [[World Model]]
- [[双臂机器人]]
- [[VLA]]
