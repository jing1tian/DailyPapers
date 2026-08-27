---
type: concept
aliases: [Goal-Conditioned Behavioral Cloning]
---

# GCBC

## 定义
Goal-Conditioned Behavioral Cloning：在目标状态条件下的行为克隆，通过从 goal 到当前 state 的回溯标注构造 goal-conditioned 训练对。

## 数学形式
$$\pi_\theta = \arg\min_\theta \mathbb{E}_{(s,a,g)\sim\mathcal{D}} \left[ -\log \pi_\theta(a \mid s, g) \right]$$

其中 goal $g$ 通常取轨迹末尾状态，训练数据通过 hindsight relabeling 生成。

## 核心要点
1. 利用 Hindsight Experience Replay (HER) 思想，把轨迹终点重标注为 goal
2. 不需要额外奖励信号，纯模仿学习框架
3. 泛化到新 goal 依赖 representation 的质量

## 代表工作
- 作为 goal-conditioned offline RL 的 baseline 出现在多篇 WM 规划论文中

## 相关概念
- [[GCIQL]]
- [[JEPA]]
- [[LeFlow]]
