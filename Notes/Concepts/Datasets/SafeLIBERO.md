---
type: concept
aliases: [Safe-LIBERO]
---

# SafeLIBERO

## 定义
在 [[LIBERO]] / [[Robosuite]] 基础上构建的安全关键操作 benchmark，覆盖 Spatial/Goal/Object/Long 四个任务套件，每个任务额外设置 Level I（近距离障碍）和 Level II（更远但沿运动路径的障碍）两级障碍干扰难度，用于联合评估任务成功率与避碰安全性。

## 核心要点
1. 沿用 LIBERO 的四个任务套件结构，但在场景中加入静态障碍物，要求策略在完成任务的同时主动避碰
2. 训练阶段仅提供 Level I 难度的演示数据，Level II 用于零样本泛化评估，考察学到的安全行为能否迁移到未见过的障碍配置
3. 配套指标包括 TSR（任务成功率）、SSR（安全成功率，需无碰撞完成任务）、CAR（无碰撞完成episode占比）、CSC（平均接触步数）、ETS（平均执行步数）
4. 单臂操作基于 [[Franka Emika Panda]] 7-DoF 机械臂，同时配有真实世界 Franka 部署任务作为仿真到真实的验证

## 代表工作
- [[SafeDojo]]: 在 SafeLIBERO 上训练并评测，在 Level I 和 Level II 上均取得最佳 TSR/SSR/ETS 综合表现

## 相关概念
- [[LIBERO]]
- [[Robosuite]]
- [[Constrained MDP]]
- [[SafeVLA]]
