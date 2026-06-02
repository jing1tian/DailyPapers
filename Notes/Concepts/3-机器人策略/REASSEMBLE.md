---
type: concept
aliases: [REASSEMBLE]
---

# REASSEMBLE VLA Transfer

## 定义
VLA 少样本迁移框架，通过分析参数更新的低维子空间（基元子空间）来高效适配新任务。

## 核心要点
1. 1. LoRA 微调时权重更新集中在低维子空间
2. 2. 基元感知训练产生可复用的子空间
3. 3. 减少新任务所需的演示数据量

## 代表工作
- [[Primitive Subspaces (paper)]]

## 相关概念
- [[OpenVLA]]
- [[LoRA]]
