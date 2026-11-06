---
type: concept
aliases: [Proactive Human-in-the-Loop, 主动人在回路]
---

# PHIL

## 定义
Proactive Human-in-the-Loop：让操作员预览机器人接下来预测的未来动作轨迹，在错误发生之前主动介入，同时通过偏好学习更新策略。

## 核心要点
1. 区别于传统 HitL（响应式干预）：PHIL 是预测式，在错误前介入
2. 持续查询策略预测未来 motion chunk，供操作员可视化预览
3. 操作员 takeover 期间收集偏好信号用于 preference-based post-training
4. 基于 [[BESO]] / [[Diffusion-Policy]] 类架构

## 相关概念
- [[BESO]]
- [[Diffusion-Policy]]
- [[HIL]]
- [[HG-DAgger]]
