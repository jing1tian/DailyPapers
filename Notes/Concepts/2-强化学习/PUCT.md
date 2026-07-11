---
type: concept
aliases: [Predictor + UCT, PUCT, UCT with Prior]
---

# PUCT（Predictor Upper Confidence bounds applied to Trees）

## 定义
PUCT 是 MCTS（Monte Carlo Tree Search）的一种变体，在 UCB 公式中引入策略网络的先验概率 $P(s,a)$，使搜索偏向策略预测的高概率动作，同时保持探索。AlphaGo 和 AlphaZero 使用此公式。

## 数学形式
$$\text{PUCT}(s, a) = Q(s, a) + c \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

其中 $Q(s,a)$ 是动作价值估计，$P(s,a)$ 是策略先验，$N(s,a)$ 是访问次数，$c$ 是探索系数。

## 核心要点
1. 与标准 UCB1 的区别：用策略先验 $P(s,a)$ 替代纯访问次数引导
2. 策略先验来自 VLA/policy 网络，决定搜索重点
3. V-VLAPS 在此基础上加入显式 value function 取代纯策略引导
4. 适合 action space 大、branching factor 高的场景

## 代表工作
- [[AlphaGo]]: 首次大规模应用 PUCT
- [[V-VLAPS]]: 用 value function 增强 PUCT for VLA planning

## 相关概念
- [[MCTS]]
- [[V-VLAPS]]
