---
type: concept
aliases: [有意识学习, 语言条件学习]
---

# Conscious Learning

## 定义

在语言描述的事件条件下，让模型学习稀疏的、语义有意义的状态转移，模拟人类通过语言理解因果事件获得高层认知的过程。

## 数学形式

$$
S_{t+\Delta} \sim p_\Theta^c(S_{t+\Delta} \mid S_t,\, z_t,\, e_{t+\Delta})
$$

其中 $e_{t+\Delta}$ 为语言事件描述。双向损失：

$$
\mathcal{L}_{\text{evt}} = \frac{1}{2}\,\mathbb{E}\left[\ell_{\text{lat}}\left(\hat{v}^l_{\text{prev}},\, v^l_{\text{prev}}\right) + \ell_{\text{lat}}\left(\hat{v}^l_{\text{next}},\, v^l_{\text{next}}\right)\right]
$$

## 核心要点

1. 以语言事件为条件，学习语义驱动的状态变化
2. 双向预测：同时预测事件前帧和事件后帧
3. 配合 VQA 损失强化语义对齐

## 代表工作

- [[Orca]]: 将有意识学习定义为语言条件双向状态转移预测，通过可学习查询 $Q_2$ 实现

## 相关概念

- [[Unconscious Learning]]
- [[Next-State Prediction]]
- [[Language-Conditioned]]
- [[Event-Conditioned Loss]]
