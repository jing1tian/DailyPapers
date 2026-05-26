---
type: concept
aliases: [HumanoidBench Benchmark]
---

# HumanoidBench

## 定义
专为全尺寸类人机器人设计的连续控制 benchmark，基于 MuJoCo/MJX，包含运动（走、跑、爬楼梯）和操控（抓取、搬运）任务，测试高维连续控制算法。

## 核心要点
1. 类人机器人模型（~21个自由度），状态空间维度远高于 DMC
2. 任务包含 locomotion（standing、walking、running）和 manipulation（pushing、lifting）
3. 支持 GPU 并行仿真（MJX），加速采样效率测试
4. 专门测试高维度、接触丰富场景下的 RL 算法
5. Dream-MPC 等论文用 HumanoidBench 验证在复杂身体下的 MPC 效果

## 代表工作
- [[Dream-MPC]]: 在 HumanoidBench 测试梯度 MPC 的泛化能力

## 相关概念
- [[DMC]]
- [[MuJoCo]]
- [[强化学习]]
