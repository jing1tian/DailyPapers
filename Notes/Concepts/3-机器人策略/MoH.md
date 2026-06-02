---
type: concept
aliases: [MoH, mixture of horizons, action chunking]
---

# Mixture of Horizons

## 定义
VLA 动作分块（action chunking）中的混合视野方法：同时维护多个 horizon 的动作头，用门控机制自适应选择。

## 核心要点
1. 1. 不同任务需要不同 action chunk length
2. 2. MoE 风格门控动态选 horizon
3. 3. 消除手动调参 horizon 超参

## 代表工作
- [[MoH (paper)]]

## 相关概念
- [[VLA]]
- [[CogACT]]
