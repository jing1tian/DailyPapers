---
type: concept
aliases: [ESD, Erased Stable Diffusion]
---

# ESD

## 定义

ESD（Erased Stable Diffusion）是一种通过微调从扩散模型中抹除特定概念的方法，用于安全对齐（消除有害内容、版权内容等）。

## 核心要点

1. 通过最小化特定概念的生成概率来抹除该概念
2. 分为 ESD-x（抹除全部相关知识）和 ESD-u（只改 unconditional 部分）两种变体
3. 是扩散模型概念消除（concept erasure）领域的代表方法
4. [[DriftScope]] 分析了 ESD 微调时产生的副作用（无意中影响其他概念）

## 相关概念

- [[SD]]
- [[DreamBooth]]
- [[DriftScope]]
