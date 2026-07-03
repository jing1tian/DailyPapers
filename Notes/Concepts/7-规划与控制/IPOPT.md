---
type: concept
aliases: [IPOPT, Interior Point OPTimizer]
---

# IPOPT

## 定义

IPOPT（Interior Point OPTimizer）是一种求解大规模非线性优化问题的开源软件，广泛用于机器人运动规划、轨迹优化等场景。

## 核心要点

1. 基于内点法（Interior Point Method）求解非线性规划（NLP）
2. 支持稀疏雅可比和海森矩阵，适合大规模问题
3. 在机器人路径规划（如 [[PVWM-Path]]）和 MPC 中常用
4. 与 CasADi 等符号计算框架配合使用

## 相关概念

- [[MPC]]
- [[PVWM]]
