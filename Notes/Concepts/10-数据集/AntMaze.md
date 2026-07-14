---
type: concept
aliases: [Ant Maze, AntMaze Benchmark]
---

# AntMaze

## 定义
基于 MuJoCo Ant 机器人的离线强化学习 benchmark，要求 ant 在迷宫中导航到目标位置，是测试长 horizon 稀疏奖励规划算法的标准环境。

## 核心要点
1. 观测：ant 关节状态 + 迷宫位置（无 RGB）
2. 奖励：稀疏，仅在到达目标附近时给出
3. 提供三种难度：umaze、medium、large
4. D4RL 数据集的重要组成部分

## 代表工作
- [[DD]]：Diffuser 在 AntMaze 上的轨迹规划测试
- [[IQL]]：隐式 Q-learning 在 AntMaze 上的基准结果

## 相关概念
- [[MuJoCo]]
- [[D4RL]]
