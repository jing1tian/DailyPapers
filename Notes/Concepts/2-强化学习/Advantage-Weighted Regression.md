---
type: concept
aliases: [AWR, advantage-weighted regression]
---

# Advantage-Weighted Regression

## 定义
一种离线强化学习方法，通过对 BC 损失加 advantage 权重，使策略优先模仿高收益动作而淡化低收益动作的影响。

## 数学形式
$$\mathcal{L}_\text{AWR} = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \exp\!\left(\frac{A^\pi(s,a)}{\beta}\right) \log \pi(a|s) \right]$$

其中 $A^\pi(s,a)$ 为 advantage 函数，$\beta$ 为温度系数。

## 核心要点
1. 不需要与环境交互，纯离线训练
2. Advantage 权重使策略自动忽略次优演示
3. 温度 $\beta$ 控制对 advantage 的敏感程度

## 代表工作
- [[CounterAlign]]: 与 counterfactual 监督结合
- [[IQL]]: 常用于计算 advantage 的离线 Q 学习方法

## 相关概念
- [[IQL]]
- [[Behavior Cloning]]
- [[Offline RL]]
