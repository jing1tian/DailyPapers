---
type: concept
aliases: [Sigmoid 路由, Sigmoid Gate, 独立路由评分]
---

# Sigmoid Router（Sigmoid 路由器）

## 定义

MoE 路由机制的一种，用 Sigmoid 函数（而非 Softmax）计算 token 与各专家之间的亲和度得分，使各专家独立评分，避免 Softmax 的归一化竞争效应。

## 数学形式

$$
\alpha_{t,j} = \text{Sigmoid}\left(u_t^\top r_j\right)
$$

- $u_t$：token $t$ 的隐藏状态（FFN 输入）
- $r_j \in \mathbb{R}^d$：专家 $j$ 的可学习路由 embedding
- $\alpha_{t,j}$：token $t$ 对专家 $j$ 的亲和度（值域 $(0,1)$，各专家独立）

top-K 选择后，门控权重归一化：

$$
g_{t,j} = \frac{\alpha_{t,j}}{\sum_{k \in \mathcal{R}_b(u_t)} \alpha_{t,k}}
$$

## 核心要点

1. **独立评分**: Sigmoid 不归一化，各专家得分互不影响，模型可以同时激活多个高亲和度专家
2. **对比 Softmax**: Softmax 路由中专家之间存在竞争（提高一个分数会降低其他专家的相对概率）；Sigmoid 完全解耦
3. **配合偏置修正**: Sigmoid 亲和度与在线偏置 $b_j$ 叠加用于 top-K 选择（偏置不影响最终门控值）
4. **来自 [[DeepSeekMoE]]**: 该路由设计在 DeepSeek-V2/V3 中被验证有效

## 代表工作

- [[LingBot-Video]]: 采用 Sigmoid Router + 在线偏置校正实现视频 DiT MoE 的稳定路由
- [[DeepSeekMoE]]: 原始提出并验证 Sigmoid 路由的有效性

## 相关概念

- [[Mixture-of-Experts]]
- [[Sparse MoE]]
- [[Load Balancing]]
