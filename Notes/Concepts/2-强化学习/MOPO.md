---
type: concept
aliases: [Model-based Offline Policy Optimization]
---

# MOPO

## 定义
基于模型的离线策略优化算法，使用悲观 reward shaping 来约束策略不进入世界模型不确定的区域，核心思想是 "penalize uncertainty"。

## 数学形式
MOPO 在模型 rollout 上使用惩罚奖励：
$$\tilde{r}(s, a) = r(s, a) - \lambda \cdot u(s, a)$$
其中 $u(s, a)$ 是模型集成的不确定性估计（如 epistemic uncertainty）。

## 核心要点
1. **悲观原则**：奖励中减去不确定性惩罚，策略自然避开 OOD 区域
2. **集成不确定性**：用多个世界模型 ensemble 估计认知不确定性
3. **vs [[COMBO]]**：MOPO 是 reward shaping，COMBO 是 Q-value 约束

## 代表工作
- [[WOMBET]]: 对比 MOPO 作为基线方法

## 相关概念
- [[COMBO]]
- [[SAC]]
- [[MBRL]]
