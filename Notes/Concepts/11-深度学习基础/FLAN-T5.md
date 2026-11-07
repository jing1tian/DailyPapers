---
type: concept
aliases: [FLAN-T5, flan-t5, FLAN T5]
---

# FLAN-T5

## 定义
Google 发布的基于指令微调（Instruction Fine-tuning）的 T5 语言模型系列，在海量任务上进行 FLAN（Finetuned Language Models）式微调，具有较强的零样本和少样本语言理解能力，常用于机器人策略中作为冻结语言编码器。

## 核心要点
1. **基础架构**: 基于 [[T5]] 的编码器-解码器架构，常用其编码器部分提取语言特征
2. **指令微调**: FLAN 方法在上千个任务上进行指令格式化微调，提升语言泛化能力
3. **尺寸选择**: FLAN-T5-large（约 0.341B）是机器人策略中常见的平衡选择，计算代价低
4. **条件注入方式**: 通常冻结编码器，提取指令嵌入后通过 Cross-Attention 注入策略 Transformer

## 数学形式
$$
e_l = \text{FLAN-T5-Enc}(l), \quad e_l \in \mathbb{R}^{L \times d}
$$

其中 $l$ 为自然语言指令，$L$ 为 token 长度，$d$ 为隐层维度。

## 相关概念
- [[T5]]: 基础架构
- [[Cross-Attention]]: 语言特征注入策略网络的常用方式
- [[UMT5]]: 另一个常用于机器人策略的 T5 变体（如 Fast-WAM 使用 UMT5-XXL）

## 代表工作
- [[La-JEPA]]: 使用冻结 FLAN-T5-large（0.341B）编码语言指令，通过 cross-attention 注入 World DiT 和 Action DiT
