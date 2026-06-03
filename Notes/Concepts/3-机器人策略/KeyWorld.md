---
type: concept
aliases: [KeyWorld]
---

# KeyWorld

## 定义
SKIP 论文提出的稀疏关键帧 embodied world model，通过选取语义关键帧代替逐帧生成，降低推理开销。

## 核心要点
1. 关键帧选择器用 MaxSemDist 找语义变化最大的帧
2. 插值模块在关键帧间生成中间视频帧
3. 两阶段解耦避免每步都运行完整扩散模型

## 数学形式
$$K^* = \arg\max_{K \subset F} \sum_{k \in K} \text{SemDist}(f_k, f_{k-1})$$

## 代表工作
- [[SKIP]]: KeyWorld 是 SKIP 提出的模型名称

## 相关概念
- [[MaxSemDist]]
- [[SKIP]]
