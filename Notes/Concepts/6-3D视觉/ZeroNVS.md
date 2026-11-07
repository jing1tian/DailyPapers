---
type: concept
---

# ZeroNVS

## 定义
零样本单图新视角合成方法，用扩散先验在无多视图数据的情况下生成任意视角图像。

## 数学形式
$$\hat{I}_{novel} = \text{Diffusion}(I_{ref}, R_{target})$$

## 核心要点
1. MoMaSplat 的 NVS 对比基线
2. 依赖扩散模型先验，不保证几何一致性

## 代表工作
- 见关联笔记

## 相关概念
- [[VGGT]]
- [[MoMaSplat]]
