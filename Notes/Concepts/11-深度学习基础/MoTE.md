---
type: concept
aliases: [Mixture-of-Transformer Experts, 混合Transformer专家]
---

# MoTE（Mixture-of-Transformer Experts）

## 定义

在 Transformer 的混合注意力（Mixed Attention）之后插入的混合专家层，包含每个模态各自的**私有专家**和跨模态共享的**共享专家**，通过 top-1 路由实现视频与动作的联合建模。

## 数学形式

$$
\mathbf{w} = \operatorname{TopKSoftmax}(W_r \operatorname{LN}(\mathbf{H}))
$$

$$
\tilde{\mathbf{H}} = \mathbf{H} + \alpha \sum_{e \in \mathcal{E}} w_e E_e(\operatorname{LN}(\mathbf{H}))
$$

- $\alpha = 0.1$：残差缩放系数
- $\mathcal{E}$：激活专家集合（1 个私有专家 + 1 个共享专家）

## 核心要点

1. **位置**：插在混合注意力之后，而非替换注意力或 FFN
2. **专家结构**：每个模态 2 个私有专家 + 1 个全局共享专家（K=1 路由）
3. **平衡损失**：辅助 $\mathcal{L}_{\mathrm{balance}}$ 防止专家路由崩溃
4. **与 FFN-MoE 对比**：不替换 FFN，而是在注意力之后额外添加，保留原有 FFN 计算

## 代表工作

- [[LAW]]: 首次提出 MoTE，用于 WAM 的视频-动作联合专家建模

## 相关概念

- [[Mixture of Experts]]
- [[混合专家]]
- [[Top-K 路由]]
- [[World Action Model]]
