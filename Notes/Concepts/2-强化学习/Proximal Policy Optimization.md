---
type: concept
aliases: [PPO, 近端策略优化]
---

# Proximal Policy Optimization

## 定义

Proximal Policy Optimization（PPO）是一种 on-policy 策略梯度算法，通过限制新旧策略的比率（clip 或 KL 惩罚）来稳定训练，是当前 LLM RLHF 的主流算法之一。

## 数学形式

**PPO-Clip 目标**：

$$
\mathcal{L}_{\mathrm{PPO}}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t A_t,\; \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t\right)\right]
$$

其中 $r_t = \pi_\theta(a_t|s_t) / \pi_{\mathrm{old}}(a_t|s_t)$，$\varepsilon$ 通常为 0.1–0.2。

**KL 惩罚变体**：

$$
\mathcal{L}(\theta) = \mathbb{E}[r_t A_t] - \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{old}})
$$

## 核心要点

1. Clip 变体简单高效，是大多数实现的默认选择；
2. 配合 [[Generalized Advantage Estimation|GAE]] 使用可同时优化偏差-方差权衡；
3. CaPtain 用 token-ratio mask（而非 clip）：丢弃 $r \notin (0.2, 4)$ 的 token，行为不同于 clip；
4. 常与 [[Supervised Fine-Tuning|SFT]] 参考策略的 [[KL Divergence|KL 惩罚]] 配合防止 reward hacking。

## 代表工作

- [[CaPtain]]: 采用 PPO 变体，用 token ratio mask 替代 clip，KL 惩罚到 SFT 参考
- InstructGPT: PPO + RLHF 对齐 LLM 的标志性工作
- Turn-PPO, ArCHer: 多轮对话场景的 PPO 扩展

## 相关概念

- [[Reinforcement Learning]]
- [[Group Relative Policy Optimization]]
- [[Generalized Advantage Estimation]]
- [[KL Divergence]]
