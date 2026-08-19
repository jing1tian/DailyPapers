---
type: concept
aliases: [全身控制, WBC, Whole Body Control, 全身协调]
---

# Whole-Body Control（全身控制）

## 定义

Whole-Body Control 是一类针对具有腿部运动、腰部姿态和臂部操作等多组自由度的机器人（尤其是人形机器人），在统一框架下同时协调所有关节以完成复合任务的控制/策略方法。

## 核心要点

1. **高维度耦合**: 人形机器人典型自由度 30+，腿部稳定性、腰部柔顺性与臂部精度存在强耦合
2. **时序依赖**: locomotion → waist stability → arm manipulation 在动力学上有天然优先序
3. **任务分解策略**: 可通过优先级任务空间（Hierarchical Task Space）或分层生成（如 HAF）解耦各子系统
4. **挑战**: 全身动作空间下单一策略学习样本效率低，端到端策略难以维持运动学一致性

## 代表工作

- [[HAF]]: 通过三阶段分层流匹配去噪实现人形机器人全身行走-操作协调
- [[Loco-Manipulation]]: 行走与操作联合任务的通用框架

## 相关概念

- [[Loco-Manipulation]]
- [[Hierarchical Chunking]]
- [[Action Chunking]]
- [[Flow-Matching]]
