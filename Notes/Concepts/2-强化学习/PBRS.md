---
type: concept
aliases: [Potential-Based Reward Shaping, 势函数奖励塑型]
---

# PBRS

## 定义
Potential-Based Reward Shaping：通过势函数 $\Phi(s)$ 对原始奖励进行增量修正 $F(s,a,s') = \gamma\Phi(s') - \Phi(s)$，保证最优策略不变。

## 数学形式
$$r'(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s)$$

## 核心要点
1. Ng et al. 1999 证明：只要 shaping term 形如 $\gamma\Phi(s') - \Phi(s)$，不改变最优策略
2. 势函数常设计为进度信号（如距 goal 距离）
3. [[TwinBridge]] 用有序事件完成记录作为离散势函数，做数字孪生 RL 的奖励迁移

## 代表工作
- [[TwinBridge]]: digital twin 场景下用 PBRS 做 sim→real 奖励迁移

## 相关概念
- [[ReWiND]]
- [[SERL]]
