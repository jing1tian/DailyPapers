---
type: concept
aliases: [残差适配器, Adapter Tuning, Parameter-Efficient Adapter]
---

# Residual Adapter（残差适配器）

## 定义

在冻结的预训练模型每层之后插入可训练的低秩瓶颈模块，通过残差连接将适配输出叠加到原始特征上，实现参数高效的领域迁移。

## 数学形式

$$
h_l^+ = h_l + \alpha_l \, W_{\text{up}}^{(l)} \, \sigma\!\left(W_{\text{down}}^{(l)} \, \text{LN}(h_l)\right)
$$

其中 $W_{\text{down}} \in \mathbb{R}^{r \times d}$（降维），$W_{\text{up}} \in \mathbb{R}^{d \times r}$（升维），$r \ll d$，$\alpha_l$ 为块级缩放系数。

## 核心要点

1. **冻结主干**：预训练权重不更新，只有适配层参数参与梯度计算
2. **低秩瓶颈**：通过降维-激活-升维压缩参数量（通常 $r = 64 \sim 256$）
3. **残差保护**：残差连接使初始化时适配输出近零，避免破坏预训练表征
4. **每层独立**：可为不同层设不同秩，灵活分配参数预算

## 代表工作

- [[DECOWAM]]: 冻结 FastWAM 骨干，用残差适配器在每个 Transformer 块后注入机器人领域知识（25.95M 可训练参数）
- Houlsby et al. (2019) "Parameter-Efficient Transfer Learning for NLP"：原始 Adapter 论文
- [[LoRA]]: 另一种低秩适配思路，作用于权重矩阵而非激活值

## 相关概念

- [[LoRA]]
- [[知识蒸馏|Knowledge Distillation]]
