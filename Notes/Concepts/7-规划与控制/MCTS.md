---
type: concept
aliases: [Monte Carlo Tree Search, 蒙特卡洛树搜索]
---

# MCTS

## 定义
Monte Carlo Tree Search，一种基于随机采样的启发式树搜索算法，通过 Selection、Expansion、Simulation、Backpropagation 四步迭代构建搜索树，广泛用于游戏 AI 和规划。

## 数学形式
UCB1 选择公式：
$$a^* = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right]$$

PUCT（带先验的 UCB，AlphaGo 风格）：
$$a^* = \arg\max_a \left[ Q(s,a) + c \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)} \right]$$

## 核心要点
1. Selection：从根节点按 UCB 策略选到叶节点
2. Expansion：对叶节点展开一个或多个子节点
3. Simulation：从展开节点随机 rollout 到终局
4. Backpropagation：将 reward 沿路径反传更新 Q 值
5. AlphaGo/AlphaZero 用神经网络 policy 和 value 替代随机 rollout

## 代表工作
- [[AlphaGo]]: 首次将深度网络与 MCTS 结合用于围棋
- [[V-VLAPS]]: 在 VLA 规划中用 MCTS + 显式值函数替代 policy prior

## 相关概念
- [[PUCT]]
- [[Model Predictive Control]]
- [[强化学习]]
