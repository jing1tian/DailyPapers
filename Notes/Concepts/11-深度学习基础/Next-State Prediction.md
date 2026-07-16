---
type: concept
aliases: [下一状态预测, NSP, Next State Prediction]
---

# Next-State Prediction

## 定义

一种统一的世界建模范式，将预测目标从 next-token/next-frame/next-action 泛化为对通用世界隐状态 $S_{t+\Delta}$ 的预测，使模型能同时支持语言、视觉和动作等多模态下游任务。

## 数学形式

$$
S_{t+\Delta} \sim p_\Theta(S_{t+\Delta} \mid S_t,\, z_t,\, c_t), \quad \Delta \in \mathbb{Z}_{\neq 0}
$$

其中 $z_t$ 为隐式动态，$c_t$ 为显式条件（语言/事件，可为空）

## 核心要点

1. 不针对特定输出模态设计，学习与模态无关的世界状态表示
2. 支持前向预测（$\Delta > 0$）和后向推断（$\Delta < 0$）
3. 下游任务通过轻量 Readout 头接入，无需微调骨干

## 代表工作

- [[Orca]]: 提出 NSP 范式的通用世界基础模型，通过无意识 + 有意识双路预训练实现

## 相关概念

- [[World Latent Space]]
- [[World State Transition]]
- [[Unconscious Learning]]
- [[Conscious Learning]]
- [[V-JEPA]]
