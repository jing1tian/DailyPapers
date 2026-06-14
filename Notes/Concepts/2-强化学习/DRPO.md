---
type: concept
aliases: [DRPO, Divergence Regularization Policy Optimization]
---

# DRPO

## 定义
LLM 强化学习中的散度正则化策略优化方法，重新审视 GRPO/PPO 中 KL 散度正则化的必要性与替代方案。

## 核心要点
1. 分析 token-level MDP 中不同散度度量（KL、TV、JS 等）对训练稳定性的影响
2. 提出在 off-policy LLM RL 中用更合适的散度替代 KL，缓解 training-inference mismatch
3. 与 [[GRPO]]、[[PPO]]、DAPO、SPO 等方法对比

## 数学形式
$$
\mathcal{L}_{\text{DRPO}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}\left[A(x,y) \cdot \log\pi_\theta(y|x) - \beta \cdot D({\pi_\theta} \| \pi_{\text{ref}})\right]
$$
其中 $D$ 为所选散度度量（非必须是 KL）。

## 代表工作
- "Rethinking the Divergence Regularization in LLM RL" (arXiv 2506.09821)

## 相关概念
- [[GRPO]]
- [[2-强化学习]]
