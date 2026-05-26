---
type: concept
aliases: [Semi-Markov Decision Process, 半马尔可夫决策过程]
---

# SMDP

## 定义
Semi-Markov Decision Process，MDP 的扩展，允许动作（或 option）执行多个时间步后才转移状态，用于建模 temporal abstraction 和 action chunking。

## 数学形式
SMDP 中动作 $a$ 持续 $\tau_a$ 步：
$$V(s) = \max_a \left[ r(s,a) + \gamma^{\tau_a} \sum_{s'} P(s'|s,a) V(s') \right]$$

## 核心要点
1. 与标准 MDP 的区别：动作执行时间 $\tau_a$ 是随机变量
2. Option 框架：每个 option 是一个子策略 + 终止条件
3. Action chunking（如 VLA 的动作块）可以用 SMDP 形式化建模
4. 折扣因子需要根据 chunk 长度调整：$\gamma^{\tau}$

## 代表工作
- [[VGAS]]: 用 SMDP 框架形式化 VLA 的 action chunk 选择问题

## 相关概念
- [[Action Chunking]]
- [[强化学习]]
- [[CQL]]
