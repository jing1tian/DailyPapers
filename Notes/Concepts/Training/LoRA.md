---
type: concept
aliases: [Low-Rank Adaptation, 低秩适应]
---

# LoRA

## 定义
Low-Rank Adaptation：一种参数高效微调方法，通过在预训练权重矩阵上附加低秩分解的增量矩阵来适配下游任务，大幅减少可训练参数数量。

## 数学形式
$$
W' = W_0 + \Delta W = W_0 + BA
$$

其中 $W_0 \in \mathbb{R}^{d \times k}$ 为冻结的预训练权重，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。可训练参数量从 $dk$ 降至 $r(d+k)$。

## 核心要点
1. 冻结原始权重，仅训练低秩矩阵 $A, B$，显著节省显存和计算
2. 推理时合并 $W' = W_0 + BA$，不增加推理延迟
3. $r$ 的选择在 4-64 之间，越大能力越强但参数越多
4. 广泛用于 LLM/VLM/diffusion 的轻量化微调（DreamBooth、D-OPSD 等）

## 代表工作
- [[D-OPSD]]: 用 LoRA 对 step-distilled 扩散模型做参数高效 on-policy 微调
- [[CoME]]: 使用 LoRA（rank 8-64）实现测试时微调长期记忆，隐式正则化防止灾难遗忘

## 相关概念
- [[Diffusion Model]]
- [[EMA]]
