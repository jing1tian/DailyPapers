---
type: concept
aliases: [世界状态转移, 状态转移动力学]
---

# World State Transition

## 定义

描述世界在时间维度上状态变化规律的数学模型，是世界模型的核心建模对象，包括自然物理演化（无意识）和语言条件化事件驱动变化（有意识）两类。

## 数学形式

通用形式：

$$
S_{t+\Delta} \sim p_\Theta(S_{t+\Delta} \mid S_t,\, z_t,\, c_t)
$$

无意识（纯视频）：$c_t = \emptyset$

有意识（语言条件）：$c_t = e_{t+\Delta}$（事件描述）

## 核心要点

1. 统一框架下 $\Delta$ 可正可负，支持前向预测和后向推断
2. 隐式动态 $z_t$（如惯性、重力）无法直接观测，需由模型隐式学习
3. 显式条件 $c_t$（语言事件）提供因果语义约束
4. 正确建模状态转移是 Next-State Prediction 的核心能力

## 代表工作

- [[Orca]]: 以双范式（无意识+有意识）预训练学习通用状态转移

## 相关概念

- [[World Latent Space]]
- [[Next-State Prediction]]
- [[Unconscious Learning]]
- [[Conscious Learning]]
