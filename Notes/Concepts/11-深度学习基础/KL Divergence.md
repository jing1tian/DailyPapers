---
type: concept
aliases: [KL散度, Kullback-Leibler Divergence, KL惩罚]
---

# KL Divergence

## 定义

KL 散度（Kullback-Leibler Divergence）是衡量两个概率分布 $P$ 和 $Q$ 差异程度的非对称度量，在强化学习中常用作策略正则化项，防止策略在更新时偏离参考策略过远。

## 数学形式

$$
\mathrm{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

在 LLM RL 中的 token 级估计（k3 估计量）：

$$
\mathrm{KL}_{k3}(\pi_\theta \| \pi_{\mathrm{ref}}) \approx \frac{1}{2}\left(r - 1 - \log r\right), \quad r = \frac{\pi_\theta(a)}{\pi_{\mathrm{ref}}(a)}
$$

## 核心要点

1. 非对称：$\mathrm{KL}(P\|Q) \ne \mathrm{KL}(Q\|P)$；
2. 非负：$\mathrm{KL}(P\|Q) \ge 0$，当且仅当 $P=Q$ 时等号成立；
3. RL 中用 KL 惩罚防止 reward hacking：模型在最大化奖励时不偏离 SFT 参考策略太远；
4. k3 估计量比直接计算更稳定，常用于 PPO/GRPO。

## 代表工作

- [[CaPtain]]: $\beta \cdot \mathrm{KL}_{k3}(\pi_\theta \| \pi_{\mathrm{SFT}})$ 作为策略损失正则项

## 相关概念

- [[Reinforcement Learning]]
- [[Proximal Policy Optimization]]
- [[Supervised Fine-Tuning]]
