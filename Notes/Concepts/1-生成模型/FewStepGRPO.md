---
type: concept
---

# FewStepGRPO

## 定义
针对 few-step 扩散模型的 GRPO 变体，解决平均速度场与现有 RL 目标（基于瞬时速度）之间的不匹配问题。

## 数学形式
$$\mathcal{L}_{GRPO} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \text{clip}(r_t, 1\pm\epsilon)\hat{A}_t\right)\right]$$

## 核心要点
1. 扩散模型 RL 微调的 few-step 专用算法
2. FewStepRL 论文提出

## 代表工作
- 见关联笔记

## 相关概念
- [[GRPO]]
- [[Flow-Matching]]
