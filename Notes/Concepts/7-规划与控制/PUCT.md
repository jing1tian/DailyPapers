---
type: concept
aliases: [Predictor Upper Confidence bound for Trees, PUCT]
---

# PUCT

## 定义
Predictor + Upper Confidence bound applied to Trees，AlphaGo/AlphaZero 使用的 MCTS 节点选择公式，用神经网络先验概率 $P(s,a)$ 引导探索，平衡 exploitation（Q 值）和 exploration（访问次数倒数）。

## 数学形式
$$a^* = \arg\max_a \left[ Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)} \right]$$

其中 $P(s,a)$ 是 policy 网络输出的先验概率，$N(s,a)$ 是动作访问次数。

## 核心要点
1. 相比纯 UCB1，用 policy prior $P(s,a)$ 偏向高概率动作，减少无效探索
2. V-VLAPS 的核心发现：用学出来的值函数替换 $Q(s,a)$ 项，比依赖 policy prior 更鲁棒
3. $c_{puct}$ 控制探索强度，AlphaGo 论文中设为 $\sqrt{2}$

## 代表工作
- [[V-VLAPS]]: 在 VLA 的 MCTS 中用显式值函数替代 PUCT 的 policy prior
- [[MCTS]]: PUCT 是 MCTS 的节点选择变体

## 相关概念
- [[MCTS]]
- [[强化学习]]
