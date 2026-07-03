---
type: concept
aliases: [TOLD, Task-Oriented Latent Dynamics]
---

# TOLD

## 定义

TOLD（Task-Oriented Latent Dynamics）是 TD-MPC 框架中的核心思想：在隐空间中学习与任务目标对齐的动力学模型，而非通用的像素级预测。

## 核心要点

1. 隐空间编码只保留对任务有用的信息（task-oriented）
2. 联合训练 encoder、dynamics、reward、Q 函数
3. 避免了像素重建的高维计算负担
4. 是 [[TD-MPC2]] 的基础概念

## 相关概念

- [[TD-MPC2]]
- [[MPC]]
- [[RSSM]]
