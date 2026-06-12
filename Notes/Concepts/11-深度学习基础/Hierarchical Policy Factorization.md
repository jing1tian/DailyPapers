---
type: concept
aliases: [层次策略分解, 分层策略因子化]
---

# Hierarchical Policy Factorization

## 定义
将复杂操作策略分解为多个条件分布乘积的建模方式，通过引入中间潜变量（技能）实现高层规划与低层执行的解耦，使长视野任务可以分层次逐级预测。

## 数学形式

$$
\pi(a_{t:t+K-1} | o, \ell, M) = \int p(a | Z^l, o) \cdot p(Z^l | z^h, o, M) \cdot p(z^h | o, \ell, M) \, dZ^l \, dz^h
$$

## 核心要点
1. 将单一复杂分布分解为三个相对简单的条件分布
2. 高层预测技能意图，低层展开具体动作序列
3. 记忆 M 仅在高层和低层展开时引入，保持推理的层次清晰性
4. 通过边缘化中间潜变量，训练时可以用不同监督源分别优化各层

## 代表工作
- [[HiMem-WAM]]: 核心策略分解框架

## 相关概念
- [[Hierarchical Latent Action]]
- [[Memory Gating]]
- [[Action Chunking]]
