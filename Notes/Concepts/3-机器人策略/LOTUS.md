---
type: concept
aliases: [LOTUS VLA]
---

# LOTUS

## 定义
LOTUS 是一个面向 3D 机器人操控的 VLA 框架，通过学习通用的 3D 场景表示来提升策略对新物体和新场景的泛化能力。

## 核心要点
1. 以 3D 点云作为输入，从 RGBD 数据中提取任务相关的 3D 特征
2. 专注于少样本泛化，减少对大量任务特定数据的依赖
3. 常被 BridgeVLA++ 等方法作为 3D VLA baseline

## 代表工作
- [[BridgeVLA++]]: 在 GemBench 和 COLOSSEUM 上与 LOTUS 做对比

## 相关概念
- [[VLA]]
- [[SpatialVLA]]
- [[RVT-2]]
