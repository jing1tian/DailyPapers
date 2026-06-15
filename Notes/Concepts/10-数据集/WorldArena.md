---
type: concept
aliases: [WorldArena]
---

# WorldArena

## 定义

WorldArena 是一个机器人世界模型评估基准，覆盖物理正确性、几何一致性和交互保真度三个维度，但仅针对机器人操作场景，不支持长视频评估，也不涵盖游戏和真实世界领域。

## 核心要点

1. **三维度覆盖**: 物理 + 几何 + 交互（与 WorldOlympiad 相同维度）
2. **单领域限制**: 仅限机器人场景
3. **无长视频支持**: 不评估长时序交互能力

## 与 WorldOlympiad 的对比

WorldOlympiad 在 WorldArena 基础上扩展了：
- 游戏领域（GameGen-X）
- 真实世界领域（LVD-2M）
- 长视频序列评估（最多 6 段字幕）

## 代表工作

- [[WorldOlympiad]]: 将 WorldArena 扩展为多领域、支持长视频的综合基准

## 相关概念

- [[世界模型 (World Model)]]
- [[EWMBench]]
- [[VBench]]
