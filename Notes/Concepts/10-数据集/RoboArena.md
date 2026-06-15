---
type: concept
aliases: [RoboArena Benchmark]
---

# RoboArena

## 定义

一个用于评估通用机器人策略的真实世界基准，通过在真实机器人上部署多种通用策略并记录成功率，为世界模型的策略评估代理提供对照标准。

## 核心要点

1. 包含 7 种通用机器人操作策略（如 π₀-FAST 等）
2. 在 DROID 设置（Franka Panda）上执行 65 个测试 session
3. 可用于评估虚拟世界模型预测与真实机器人执行的相关性
4. 评估指标包括：Pearson 相关系数 r、Spearman 相关系数 ρ、MMRV（Mean Max Rank Violation）

## 代表工作

- [[OSCAR]]: 在 RoboArena 上验证策略评估代理能力，Pearson r = 0.750

## 相关概念

- [[DROID]]
