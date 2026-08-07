---
type: concept
aliases: [COLOSSEUM Benchmark]
---

# COLOSSEUM

## 定义
COLOSSEUM 是一个专为测试机器人操控策略**分布外泛化**能力而设计的 benchmark，通过系统性地修改视觉属性（颜色、纹理、光照、背景）来评估策略的鲁棒性。

## 核心要点
1. 提供 20 种视觉属性的系统扰动，构造 OOD 测试集
2. 专注于策略对视觉变化的鲁棒性评估，而非任务复杂度
3. 用于 VLA 和 3D 操控策略的泛化性评估

## 代表工作
- [[BridgeVLA++]]: 使用 COLOSSEUM 评估记忆增强 3D VLA 的泛化能力

## 相关概念
- [[GemBench]]
- [[LIBERO]]
- [[RoboTwin]]
