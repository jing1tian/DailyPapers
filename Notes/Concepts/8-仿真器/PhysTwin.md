---
type: concept
aliases: [PhysTwin Dataset, Physics Twin]
---

# PhysTwin

## 定义
一个用于 real-to-sim 转换研究的数据集/框架，包含真实物理交互的视频 episodes，目标是构建物理一致的数字孪生仿真。

## 核心要点
1. 提供真实机器人与物体交互的视频数据（含物理轨迹）
2. 用于评估自动化 real2sim 流水线的对齐精度
3. 与 [[DROID]] 数据集一起作为 Agentic Real2Sim 的测试基准

## 代表工作
- [[AgenticReal2Sim]]: 使用 PhysTwin 数据做 real2sim 评测

## 相关概念
- [[MuJoCo]]（仿真目标环境）
- [[DROID]]（同类 real robot 数据集）
- [[FoundationPose]]（real2sim 中的姿态估计工具）
