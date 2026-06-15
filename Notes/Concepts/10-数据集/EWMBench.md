---
type: concept
aliases: [EWMBench, Embodied World Model Benchmark]
---

# EWMBench

## 定义

EWMBench（Embodied World Model Benchmark）是专注于机器人具身场景的世界模型评估基准，评估生成视频对动作指令的响应保真度（交互性），但不覆盖物理一致性或几何一致性。

## 核心要点

1. **单领域**: 仅评估机器人操作场景
2. **单维度**: 聚焦交互响应，不含物理/几何评估
3. **局限性**: 不支持长视频评估，缺乏多领域泛化

## 与 WorldOlympiad 的对比

WorldOlympiad 在 EWMBench 基础上扩展了：
- 物理正确性评估维度
- 3D 几何一致性评估维度
- 游戏和真实世界领域

## 代表工作

- [[WorldOlympiad]]: 将 EWMBench 扩展为多维度、多领域综合基准

## 相关概念

- [[世界模型 (World Model)]]
- [[VBench]]
- [[WorldArena]]
