---
type: concept
aliases: [DeFI, Decoupled Fine-grained Intent]
---

# DeFI

## 定义
VLA RL post-training 方法，通过细粒度意图解耦改善策略学习的稳定性，被 TEMPO 作为对比 baseline。

## 核心要点
1. 将 VLA 的 language understanding 和 action generation 分开做 RL 优化
2. 细粒度的意图表示有助于对齐语义和运动
3. 与 [[TEMPO]] 的语义-动作解耦思路相近但实现不同

## 相关概念
- [[TEMPO]]
- [[ConRFT]]
- [[VLA-RL]]
