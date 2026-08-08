---
type: concept
aliases: [SONIC Controller, SONIC Whole-Body Controller, 全身控制器]
---

# SONIC

## 定义

SONIC 是一个基于仿真训练的**人形机器人全身控制器**，将 64 维紧凑动作潜变量映射为机器人关节力矩，支持上肢操作与下肢运动的统一协调控制。

## 核心要点

1. **64 维动作潜变量接口**: 高层策略（如 [[World Action Model]] / VLA）输出 64 维潜变量，SONIC 将其解码为全身关节指令
2. **全身统一控制**: 不区分操作臂与行走腿，在同一潜变量空间中统一表示并发运动-操作动作
3. **仿真回放（Simulation Replay）**: 通过 SONIC 在仿真中回放人类/公开数据集动作，提取可部署的机器人动作潜变量，实现人类到机器人的先验迁移
4. **遥操作接口**: 支持基于 Pico 4 Ultra VR 头显 + 足部追踪器的遥操作数据采集，捕获全身运动参考

## 动作表示

- **动作潜变量**: 64 维（由 SONIC 定义的全身运动潜空间）
- **附加手部指令**: 2 维（0 = 张开，1 = 闭合）
- **总动作维度**: 66 维

## 代表工作

- [[Omega0]]: ω-0 使用 SONIC 作为低级控制器，Stage 1 训练 FAST Tokenizer 对 SONIC 潜变量离散化，Stage 2 学习预测 SONIC 兼容的全身动作潜变量

## 相关概念

- [[World Action Model]]
- [[Loco-Manipulation]]
- [[Real-Time Chunking]]
- [[FAST]]
