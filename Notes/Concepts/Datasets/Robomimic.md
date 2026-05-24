---
type: concept
aliases: [Robomimic, robomimic benchmark]
---

# Robomimic

## 定义

Robomimic 是一个用于机器人操作模仿学习的大规模基准和框架，提供标准化的演示数据集、任务环境和评估协议。

## 核心要点

1. 包含 Square、Transport、Tool-Hang 等多个难度递增的操作任务
2. 提供不同质量（专家/人类）的演示数据，支持多种学习算法对比
3. 基于 MuJoCo 仿真，支持多摄像头观测
4. OOD 评估通常通过扰动机器人初始关节状态实现

## 代表工作

- [[FeedbackWM]]: 在 Robomimic Square/Transport/Tool-Hang 上评估 OOD 操作性能

## 相关概念

- [[Diffusion Policy]]
- [[World Model]]
