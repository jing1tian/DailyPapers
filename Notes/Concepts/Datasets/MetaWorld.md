---
type: concept
aliases: [MetaWorld, Meta-World]
---

# MetaWorld

## 定义
UC Berkeley 发布的机器人操作 benchmark，基于 MuJoCo 物理引擎，包含 50 种桌面操作任务，支持多任务学习和元学习研究。

## 核心要点
1. 50 个操作任务（开抽屉、堆叠、推块等），统一的状态/动作空间
2. 支持 MT10、MT50 等多任务 benchmark 设置
3. 常用于评估任务泛化和持续学习能力
4. ML1/ML10/ML45 用于元学习评估

## 代表工作
- Yu et al. (2019): "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"

## 相关概念
- [[MuJoCo]]
- [[LIBERO]]
- [[RoboCasa]]
