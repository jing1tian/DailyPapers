---
type: concept
aliases: [预训练总损失, 多目标预训练损失]
---

# Pre-training Loss

## 定义

多任务预训练中各学习目标的加权线性组合，在 Orca 中由无意识学习损失、有意识学习损失和 VQA 损失三项构成。

## 数学形式

$$
\mathcal{L} = \lambda_{\text{obs}} \cdot \mathcal{L}_{\text{obs}} + \lambda_{\text{evt}} \cdot \mathcal{L}_{\text{evt}} + \lambda_{\text{vqa}} \cdot \mathcal{L}_{\text{vqa}}
$$

Orca 使用的权重：$\lambda_{\text{obs}}=0.1,\; \lambda_{\text{evt}}=0.5,\; \lambda_{\text{vqa}}=0.4$

## 核心要点

1. 权重比例反映各目标的相对重要性：有意识学习 > 语义对齐 > 无意识学习
2. 消融实验验证三个目标缺一不可，组合效果最优
3. 类似设计广泛见于多模态预训练（如 CLIP + 重建 + 生成的混合目标）

## 代表工作

- [[Orca]]: 三目标组合预训练

## 相关概念

- [[Observation-Only Loss]]
- [[Event-Conditioned Loss]]
- [[VQA]]
