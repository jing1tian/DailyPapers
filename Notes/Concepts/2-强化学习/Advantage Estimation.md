---
type: concept
aliases: [优势估计, Advantage Function, A(s,a)]
---

# Advantage Estimation

## 定义
优势函数 $A(s,a) = Q(s,a) - V(s)$ 衡量在状态 $s$ 下执行动作 $a$ 相对于平均水平的额外价值，是策略梯度方法的核心组件。

## 数学形式

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

广义优势估计（GAE）：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## 核心要点
1. 优势函数通过减去基线 $V(s)$ 降低方差，同时保持策略梯度无偏
2. GAE 用 $\lambda$ 平衡偏差（小 $\lambda$）与方差（大 $\lambda$）
3. 在模型辅助的策略提升中，优势可在想象轨迹上估计，无需真实交互

## 代表工作
- [[WEAVER]]: 在世界模型想象轨迹上计算优势，筛选高质量合成数据微调策略

## 相关概念
- [[TD(λ)]]
- [[流匹配]]
- [[世界模型]]
