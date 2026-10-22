---
type: concept
aliases: [Diversity is All You Need, 无监督技能发现]
---

# DIAYN

## 定义
一种无监督技能发现方法，通过最大化技能与状态之间的互信息来学习多样化、可区分的技能策略，无需任何外部奖励。

## 数学形式
$$\mathcal{F}(\theta) = H[A|S,Z] - H[A|S] + I(S;Z)$$
其中 $Z$ 是离散技能变量，$I(S;Z)$ 是状态与技能的互信息，用判别器 $q_\phi(z|s)$ 近似：
$$\mathcal{F}(\theta) \approx \mathbb{E}[\log q_\phi(z|s) - \log p(z)]$$

## 核心要点
1. 无需任务奖励，只靠内在奖励学习技能多样性
2. 用判别器区分不同技能产生的状态分布
3. 学到的技能可作为 downstream 任务的基础策略
4. 天然适合作为 world model 探索的内在奖励基础

## 代表工作
- [[DIAYN]]: Eysenbach et al., 2019 (原始论文)

## 相关概念
- [[AWR]]
- [[PPO]]
- [[METRA]]
