---
type: concept
aliases: [Selective Parallel Attention with Residual Connections]
---

# SPARC

## 定义
一种门控跨模态注意力机制，用于在主干网络中选择性融合异步多模态信号。

## 核心要点
1. 通过门控（gating）控制高频模态信号何时注入主干
2. 与 GCA（Gated Cross Attention）思路相近
3. 用于 DAM-VLA 中实现异步多模态融合

## 数学形式
$$h_{out} = h_{main} + g \cdot 	ext{CrossAttn}(h_{main}, x_{modal})$$

其中 $g$ 是学习的门控标量。

## 代表工作
- [[DAM-VLA]]: 用 SPARC 实现多模态异步 VLA

## 相关概念
- [[Attention Pooling]]
- [[Co-training]]
