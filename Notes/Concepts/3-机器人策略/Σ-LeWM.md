---
type: concept
aliases: [Sigma-LeWM, S-Information World Model, LeWM-Sigma]
---

# Σ-LeWM

## 定义
基于 S-Information（Σ 信息）的 JEPA 式潜在世界模型学习框架，用 S-Information 目标替代 pixel reconstruction loss，在机器人控制任务中学习更丰富的世界模型表示。

## 核心要点
1. S-Information 是信息论度量，同时捕获局部和全局表示结构
2. 基于 [[JEPA]] 架构（Joint Embedding Predictive Architecture）
3. DCL（Decoupled Contrastive Loss）做表示正则化
4. 在 Gear-Assembly 等精细操作任务上验证（真实实验）
5. Vrije Universiteit Amsterdam 出品

## 代表工作
- [[Σ-LeWM]]: 原始论文（VU Amsterdam，2611.12732）
- [[LeWM]]: 前序工作

## 相关概念
- [[JEPA]]
- [[LeWM]]
- [[CEM]]
