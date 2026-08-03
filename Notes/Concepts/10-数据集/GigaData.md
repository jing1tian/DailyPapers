---
type: concept
aliases: [GigaData Dataset]
---

# GigaData

## 定义
GigaWorld 项目配套的大规模机器人操作数据集，专为训练和评估 world model for robot policy 而构建。

## 数学形式
$$\mathcal{D}_{\text{GigaData}} = \{(o_t, a_t, o_{t+1})\}_{t=1}^{T}$$

## 核心要点
1. 由 GigaAI / Tsinghua University 构建
2. 专注于操作任务的轨迹数据，包含视觉观察和动作对
3. 配合 [[GigaBrain]] 和 [[GigaWorld1]] 使用，作为 world model 评估基础
4. 覆盖多种操作场景和物体类别

## 代表工作
- [[GigaWorld1]]: "GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation" (2607.02642)

## 相关概念
- [[GigaWorld1]]
- [[GigaBrain]]
- [[AgiBot World]]
