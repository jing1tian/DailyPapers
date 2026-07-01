---
type: concept
aliases: [Temporal Token Merging, Temporal Me]
---

# TempMe

## 定义
针对视频 Transformer 的时序 token merging 方法，在时间维度合并相邻帧的相似 token 以降低计算量。

## 核心要点
1. 时序维度的 token 相似度计算 + 合并
2. 与 ToMe 空间合并互补，可组合使用
3. training-free，plug-and-play

## 代表工作
- [[ST-Merge]]: TempMe 的对比 baseline

## 相关概念
- [[ToMe]]
- [[视觉 Token 剪枝]]
