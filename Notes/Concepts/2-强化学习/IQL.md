---
type: concept
aliases: [Implicit Q-Learning, 隐式Q学习]
---

# IQL (Implicit Q-Learning)

## 定义
一种 offline RL 算法，通过隐式地约束策略更新（不直接查询 out-of-distribution 动作）来解决 offline RL 的外推误差问题。

## 数学形式
$$L_V(\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[L_2^\tau(Q_{\hat\theta}(s,a) - V_\psi(s))\right]$$

其中 $L_2^\tau$ 是期望回归损失（expectile loss），$\tau \in (0,1)$ 控制对高值动作的关注程度。

## 核心要点
1. 仅在数据集动作上评估 Q 函数，避免对未见动作的外推
2. 用 expectile regression 隐式地估计 $\max_a Q(s,a)$，而不真的查询新动作
3. 适合作为 offline pretraining 基础，再做 online fine-tuning（如 [[BORA]] 的框架）
4. 比 CQL、TD3+BC 等方法在多数 offline benchmark 上更简洁且有竞争力

## 代表工作
- [[BORA]]: 用 IQL 做 offline 预训练，再用在线 PPO 做残差适应
- Kostrikov et al. (2021): 原始 IQL 论文

## 相关概念
- [[强化学习]]
- [[RLPD]] — 另一种 offline+online 混合 RL 方法
