---
type: concept
aliases: [SPM, Selective Parameter Merging, Separable Concept Editing]
---

# SPM

## 定义

SPM（Separable Concept Editing / Selective Parameter Merging）是一种扩散模型概念编辑方法，通过选择性地合并/分离参数来精准控制概念的学习和消除，同时保留其他概念。

## 核心要点

1. 将目标概念的参数变化与其他概念解耦
2. 比 ESD 等全局方法有更小的副作用
3. 适用于概念学习（个性化）和概念消除（安全对齐）两种场景
4. [[DriftScope]] 分析了 SPM 微调的隐性概念漂移

## 相关概念

- [[SD]]
- [[DreamBooth]]
- [[ESD]]
- [[DriftScope]]
