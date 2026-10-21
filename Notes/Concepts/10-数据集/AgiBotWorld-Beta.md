---
type: concept
aliases: [AgiBotWorld Beta, AgiBot World数据集]
---

# AgiBotWorld-Beta

## 定义
AgiBotWorld-Beta 是一个大规模机器人操作数据集，包含 RGB-D 视频序列，用于世界模型和深度预测的预训练。

## 核心要点

1. **数据规模**: 作为深度预训练的主要来源（75% 权重），与 RoboTwin 组合约 2300 小时
2. **包含深度**: 提供同步 RGB-D 数据，可用于深度 VAE 预训练
3. **机器人操作场景**: 涵盖多种操作任务场景

## 代表工作
- [[PrismWAM]]: 使用 AgiBotWorld-Beta（75%）+ RoboTwin（25%）约 2300 小时数据进行深度 VAE 和第一阶段训练

## 相关概念
- [[Depth Prediction]]
- [[World Action Model]]
