---
type: concept
aliases: [DART world model, Discrete Autoregressive Transformer]
---

# DART

## 定义

DART（Discrete AutoRegressive Transformer）是一种基于 Transformer 的 token-based 世界模型，用于视觉强化学习，通过离散自回归建模预测下一帧的 token 序列。

## 核心要点

1. **离散自回归**: 使用 VQ tokenizer 将帧编码为离散 token，再用 Transformer 自回归预测下一帧 token
2. **Craftax-classic 性能**: 1M 步达到 55.45% Return，是 ITC 之前的重要基线之一
3. **与 IRIS 的关系**: DART 是 IRIS 框架的一种演进，改进了训练稳定性

## 代表工作

- [[ITC]]: 在 Craftax-classic 上超越 DART（72.46% vs. 55.45% Return）

## 相关概念

- [[World Model]]
- [[IRIS]]
- [[ITC]]
- [[Transformer]]
