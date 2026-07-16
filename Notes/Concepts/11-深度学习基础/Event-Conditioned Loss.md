---
type: concept
aliases: [事件条件损失, 有意识学习损失]
---

# Event-Conditioned Loss

## 定义

Orca 有意识学习阶段的训练损失，在语言事件条件下对事件前后两帧状态进行双向隐空间预测，权重最高（$\lambda=0.5$），是核心学习目标。

## 数学形式

$$
\mathcal{L}_{\text{evt}} = \frac{1}{2}\,\mathbb{E}\left[\ell_{\text{lat}}\left(\hat{v}^l_{\text{prev}},\, v^l_{\text{prev}}\right) + \ell_{\text{lat}}\left(\hat{v}^l_{\text{next}},\, v^l_{\text{next}}\right)\right]
$$

## 核心要点

1. 双向预测使模型同时学习事件的前因（prev）和后果（next）
2. 以语言事件 $e_{t+\Delta}$ 为条件，强制模型理解语义因果关系
3. 消融实验显示此损失对图像预测能力最关键（缺失时图像性能断崖）

## 代表工作

- [[Orca]]: 三路预训练损失之一（$\lambda_{\text{evt}}=0.5$）

## 相关概念

- [[Conscious Learning]]
- [[Latent Matching Loss]]
- [[Observation-Only Loss]]
- [[Pre-training Loss]]
