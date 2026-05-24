---
type: concept
aliases: [Stochastic Transformer-based wOrld Model]
---

# STORM

## 定义
基于随机 Transformer 的视觉世界模型，在离散 latent token 空间建模随机动态，用于视觉 RL 的 model-based 规划，在 Atari 100K benchmark 上达到 SOTA 级别。

## 数学形式
$$z_{t+1} \sim q_\phi(z_{t+1} | z_t, a_t), \quad \hat{o}_t = \text{dec}(z_t)$$

## 核心要点
1. 用随机 Transformer 建模 latent 动态中的不确定性
2. 支持 imagination-based 规划和 RL 训练
3. 相比 DreamerV3 在部分 benchmark 上更高效
4. token-based 表示，与 IRIS 类似但引入随机性

## 代表工作
- [[STORM]]：He et al. 2023，Stochastic Transformer World Model
- [[ITC]]：Identifiable Token Correspondence，解决 STORM 类模型的 token 时序漂移

## 相关概念
- [[IRIS]]
- [[DreamerV3]]
- [[World Model]]
