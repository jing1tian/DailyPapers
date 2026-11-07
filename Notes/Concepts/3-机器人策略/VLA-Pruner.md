---
type: concept
---

# VLA-Pruner

## 定义
VLA 视觉 token 剪枝方法，识别并移除对动作预测贡献低的冗余视觉 token。

## 数学形式
$$\mathcal{T}' = \{t_i \in \mathcal{T} : s(t_i) > \tau\}$$

## 核心要点
1. 无需重训练的推理加速
2. 和 T3R 是同类工作但策略不同

## 代表工作
- 见关联笔记

## 相关概念
- [[T3R]]
- [[FastV]]
