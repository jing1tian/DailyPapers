---
type: concept
aliases: [WBench, World Model Benchmark, Interactive Video World Model Benchmark]
---

# WBench

## 定义

WBench 是 Fudan University 和 Meituan LongCat Team 提出的交互式视频世界模型综合评测基准，从 8 个维度系统评测世界模型在 multi-turn 交互场景下的一致性、可控性和物理合理性。

## 核心要点

1. **Multi-turn 评测**: 模拟用户连续给定动作输入，测试世界模型保持长程一致性的能力
2. **8 个评测维度**: 涵盖物理一致性、语义追踪、空间关系、动作可控性等
3. **自动化评估**: 用 CLIP 相似度、NavScore 等自动化指标，减少人工评测成本
4. **基线模型**: Matrix-Game、YUME、LingBot、LongCat 等世界模型作为对比

## 代表工作

- [[WBench]]: 完整论文笔记（Ying et al. 2026, arXiv 2605.25874）

## 相关概念

- [[World Model]]
- [[Matrix-Game]]
- [[视频生成世界模型]]
