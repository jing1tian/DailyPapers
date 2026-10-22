---
type: concept
aliases: [Maximum Entropy Temporal Representations and Actions]
---

# METRA

## 定义
一种通过最大化时序状态覆盖来学习多样化行为的无监督 RL 方法，用 Wasserstein 距离衡量技能的时序差异，适合高维连续空间的探索。

## 数学形式
$$\max_\pi \mathbb{E}\left[\sum_t d_W(s_t, s_{t+k})\right] \quad \text{s.t.} \quad H[\pi] \geq \epsilon$$
其中 $d_W$ 是 Wasserstein 距离，激励策略产生时序上远离的状态转移。

## 核心要点
1. 替代 DIAYN 的互信息目标，改用 Wasserstein 距离衡量时序状态多样性
2. 在高维连续动作空间中比 DIAYN 更稳定
3. 可作为 action-conditioned world model 可控性评估的基线
4. 被 AIM（Action Influence Maximization）引用为比较基准

## 代表工作
- METRA: Park et al., 2024

## 相关概念
- [[DIAYN]]
- [[AWR]]
