---
type: concept
aliases: [ABot-M0.6, ABot M0.6]
---

# ABot-M0.6

## 定义

ABot-M0.6 是一个泛化机器人操作策略模型，基于体现预训练（Embodied Pretraining）数据在 RoboCasa365 等 benchmark 上的强基线方法。

## 核心要点

1. 需要额外的机器人体现预训练数据（Embodied PT.）
2. 在 RoboCasa365 上整体成功率为 46.6%，此前为最强基线
3. 在 Composite Seen 任务上达到 48.3%，但在 Composite Unseen（泛化）任务上仅 7.9%

## 代表工作

- [[LoopWAM]]: 在 RoboCasa365 benchmark 上与 ABot-M0.6 对比，无需体现预训练即超越（49.5% vs 46.6%）

## 相关概念

- [[Generalist Robot Policy]]
- [[RoboCasa365]]
- [[World Action Model]]
