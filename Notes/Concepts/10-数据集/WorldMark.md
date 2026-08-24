---
type: concept
aliases: [WorldMark Benchmark]
---

# WorldMark

## 定义
一个用于评测交互式世界模型（interactive world model）的 benchmark，与 WorldScore、HarnessEval-W 并列为该领域主流评测框架之一。

## 数学形式
评测轴通常包含视觉一致性、物理合理性、时序连贯性等维度的量化指标。

## 核心要点
1. 专门针对世界模型在给定动作后的视觉预测质量进行评测
2. 在 HarnessEval-W 中作为同类 benchmark 对比
3. 与 [[WorldScore]]、[[PhyWorldBench]]、[[VideoPhy]] 共同构成 world model eval 生态

## 代表工作
- [[HarnessEval-W]]: 在综合评测中与 WorldMark 对比，提出 harness-based agentified 评测范式

## 相关概念
- [[WorldScore]]: 另一个 world model 评测 benchmark
- [[HarnessEval-W]]: 基于 agent 的 world model 评测框架
- [[EvalCrafter]]: 视频/生成模型评测框架
