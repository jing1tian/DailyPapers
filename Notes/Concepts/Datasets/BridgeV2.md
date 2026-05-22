---
type: concept
aliases: [Bridge Dataset V2, BridgeData V2]
---

# BridgeV2

## 定义
大规模开源机器人操作数据集，包含约 60k 条 WidowX 机械臂的真实操作轨迹，覆盖多种厨房/桌面任务，是 offline RL 和模仿学习的重要 benchmark。

## 核心要点
1. 约 60,000 条真实机器人轨迹，涵盖 10+ 任务类别
2. 使用 WidowX 低成本机械臂，硬件可复现
3. 多样化的视角（1-3 个摄像头）
4. 常用于评估 VLA、offline RL、跨任务迁移

## 代表工作
- [[RankQ]]：在 BridgeV2 上训练 offline 策略，然后迁移到 online 微调

## 相关概念
- [[LIBERO]]
- [[RoboCasa]]
