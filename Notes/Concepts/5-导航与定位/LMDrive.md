---
type: concept
aliases: []
---

# LMDrive

## 定义
基于大语言模型的闭环自动驾驶框架，用 Q-Former 压缩多视角传感器特征后输入冻结 LLM，实现语言指令驱动的端到端驾驶。

## 核心要点
1. Q-Former 将多视角 camera + LiDAR 压缩为少量 visual tokens
2. 冻结 LLM 骨架（LLaMA 等）做推理
3. 支持 CARLA 仿真训练和评估

## 代表工作
- [[MindDrive]]: 在 LMDrive 基础上引入 online RL 训练

## 相关概念
- [[CARLA]]
- [[DriveVLA]]
- [[MindDrive]]
