---
type: concept
aliases: [分层潜在动作, 层次化潜变量动作]
---

# Hierarchical Latent Action

## 定义
一种将机器人动作同时在多个时间粒度上进行潜变量建模的框架，通过低层运动潜变量捕获精细的运动细节，通过高层技能潜变量捕获长视野的任务语义。

## 数学形式

$$
\pi(a | o, \ell, M) = \int p(a | Z^l, o) \cdot p(Z^l | z^h, o, M) \cdot p(z^h | o, \ell, M) \, dZ^l \, dz^h
$$

其中 $z^l$ 为低层运动潜变量，$z^h$ 为高层技能潜变量。

## 核心要点
1. 低层潜变量通过光流重建学习运动先验
2. 高层技能潜变量通过分层分块（hierarchical chunking）得到
3. 技能边界通过相邻潜变量的不相似度评分检测
4. 两层表示联合监督，实现跨时间尺度的一致规划

## 代表工作
- [[HiMem-WAM]]: 提出该框架，用于长视野机器人操作

## 相关概念
- [[Action Chunking]]
- [[Hierarchical Chunking]]
- [[World Action Model]]
- [[Attention Pooling]]
- [[DPFlow]]
