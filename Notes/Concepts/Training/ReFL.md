---
type: concept
aliases: [Reward Feedback Learning, 奖励反馈学习, Reward Backpropagation]
---

# ReFL

## 定义

Reward Feedback Learning（ReFL）：通过对截断前缀 rollout 末段的单个 clean-output 预测直接反向传播可微 reward，来微调扩散模型的方法。

## 数学形式

在截断前缀（前缀无梯度执行）末段随机选择 query 状态 $z_t$，对 clean-output 预测 $\hat{x}_0 = f_\theta(z_t, t)$ 直接反向传播 reward：

$$
\mathcal{L}_{ReFL} = -\mathbb{E}_{t \sim \mathcal{U}(T/4, T)}\left[R(D(\hat{x}_0(z_t, t, c)))\right]
$$

其中 $D$ 为 latent decoder，$R$ 为可微 reward 函数。

## 核心要点

1. 只在 denoising 轨迹末段 25% 区间随机采样单个 query 状态
2. 前缀（noise → $z_t$）无梯度执行，只对最后一步的 clean-output 预测反向传播
3. 将 reward 评估与模型优化**耦合**在同一前向/后向 pass 中
4. 不执行去噪轨迹的后缀（$z_t$ 之后），也不完整解码最终图像

## 代表工作

- [[DiffusionOPSD]] (2608.24646)：将 ReFL 作为主要对比基线，指出其 reward 与优化耦合的局限性
- ReFL 原始论文（Xu et al., ImageReward）

## 相关概念

- [[RLHF]]
- [[DiffusionNFT]]
- [[DiffusionOPSD]]
- [[Reward Gradient]]
