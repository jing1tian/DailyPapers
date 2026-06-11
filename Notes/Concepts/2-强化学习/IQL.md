---
type: concept
aliases: [Implicit Q-Learning, 隐式 Q 学习]
---

# IQL（Implicit Q-Learning）

## 定义
IQL 是一种 offline RL 方法，通过分位数回归隐式地估计最优 Q 值，避免 OOD 动作查询，使得策略可以从固定数据集中稳定训练。

## 数学形式
$$\mathcal{L}_V = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ L_2^\tau (Q(s,a) - V(s)) \right]$$

其中 $L_2^\tau$ 为分位数 Huber 损失，$\tau \in (0,1)$ 控制价值估计的保守程度。

## 核心要点
1. 核心思想：将 Q 函数学习分解为 V 函数学习 + advantage 加权，无需在 OOD 动作上求最大化
2. 适合 offline RL 场景：只用已有数据集训练，不与环境交互
3. 常用于 robot manipulation 的 offline fine-tuning 和 behavior cloning + RL 结合

## 代表工作
- Kostrikov et al. (2021)：IQL 原始论文
- [[QGF]]：在 IQL 基础上做 test-time Q 梯度引导

## 相关概念
- [[BC]]
- [[Q-Learning]]
- [[Offline RL]]
