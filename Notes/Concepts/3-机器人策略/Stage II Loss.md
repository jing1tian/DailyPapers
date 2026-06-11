---
type: concept
aliases: [潜变量策略损失, HiMem-WAM Stage II Loss]
---

# Stage II Loss

## 定义
HiMem-WAM Stage II 中分层技能发现的训练目标，联合监督高层技能预测、低层潜变量预测和技能边界检测三个任务。

## 数学形式

$$
\mathcal{L}_{\text{latent}} = \lambda_h \, \text{MSE}(\hat{z}^h_t, \bar{z}^h_t) + \lambda_l \, \text{MSE}(\hat{Z}^l_{t:t+K-1}, Z^l_{t:t+K-1}) + \lambda_b \, \text{BCE}(\hat{b}_t, \bar{b}_t)
$$

## 核心要点
1. 高层技能 MSE 损失确保技能表示的判别性
2. 低层潜变量 MSE 损失保持与 Stage I 表示的一致性
3. 边界 BCE 损失驱动模型学习准确的技能切换时机
4. 三项损失权重 λ_h, λ_l, λ_b 需要平衡

## 代表工作
- [[HiMem-WAM]]: Stage II 训练目标

## 相关概念
- [[Hierarchical Latent Action]]
- [[Hierarchical Chunking]]
- [[Tokenizer Loss]]
