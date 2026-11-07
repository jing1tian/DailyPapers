---
type: concept
---

# EfficientTAM

## 定义
高效时序注意力模块，减少视频序列中时序 token 处理的计算量。

## 数学形式
$$\text{Attn}_{eff} = \text{softmax}\frac{QK^T}{\sqrt{d}} \cdot V \cdot \text{mask}$$

## 核心要点
1. T3R 的对比基线之一
2. 视频 token 裁剪加速领域的相关方法

## 代表工作
- 见关联笔记

## 相关概念
- [[FastV]]
- [[T3R]]
