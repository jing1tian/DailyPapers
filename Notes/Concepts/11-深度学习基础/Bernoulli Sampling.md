---
type: concept
aliases: [伯努利采样, 随机二值采样, Stochastic Binary Mask]
---

# Bernoulli Sampling

## 定义

伯努利采样（Bernoulli Sampling）是以伯努利分布对每个位置独立进行 0/1 随机采样的操作，常用于随机 dropout、随机掩码生成、数据增强等场景，是引入训练时随机性的基本工具。

## 数学形式

$$\gamma \sim \text{Bernoulli}(p)$$

$$P(\gamma = 1) = p, \quad P(\gamma = 0) = 1 - p$$

在训练调度中可使用时变概率：

$$\gamma_s \sim \text{Bernoulli}(p(s))$$

## 核心要点

1. **独立性**: 各位置的采样相互独立，每次前向传播产生不同随机掩码
2. **期望等价**: 期望上等价于以权重 $p$ 软加权，但每次训练步产生硬 0/1 决策
3. **调度扩展**: 概率 $p(s)$ 可随训练步数 $s$ 动态变化（如线性衰减），实现渐进式消融

## 代表工作

- [[MECo-WAM]]: 用衰减伯努利采样控制 4D 读取掩码的激活概率，从 $p=1$ 线性衰减至 $p=0$
- [[Dropout]]: 最经典的伯努利随机掩码应用（$p = 1 - \text{drop\_rate}$）

## 相关概念

- [[Knowledge Distillation]]
- [[Mixed Attention]]
- [[Flow Matching]]
