---
type: concept
aliases: [AgiBot World, AgiBotWorld Dataset]
---

# AgiBotWorld

## 定义

一个大规模真实世界机器人操作数据集，由 AgiBot 收集，涵盖多种家庭与工业操作场景，是 InternVLA-A1.5 等大规模 VLA 模型训练数据配方的重要组成部分。

## 核心要点

1. **规模**: 168M 帧真实机器人操作视频，是 InternVLA-A1.5 六个训练数据源中帧数第二大的真实数据集
2. **场景多样性**: 覆盖多种日常操作任务，具备较高的场景与动作多样性
3. **在 VLA 训练中的作用**: 与 DROID、UMI、Galaxea 等数据集共同构成真实世界训练语料库

## 数据规模

| 指标 | 数值 |
|------|------|
| 帧数 | 168M 帧 |
| 类型 | 真实世界操作数据 |
| 来源 | AgiBot（AgiBotWorld） |

## 代表工作

- [[InternVLA-A1.5]]: 使用 AgiBotWorld 作为六个机器人训练数据源之一（168M 帧）

## 相关概念

- [[DROID]]
- [[UMI]]
- [[LIBERO]]
