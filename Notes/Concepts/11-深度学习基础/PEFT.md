---
type: concept
aliases: [Parameter-Efficient Fine-Tuning, 参数高效微调]
---

# PEFT

## 定义
参数高效微调（Parameter-Efficient Fine-Tuning）：在大模型微调时只更新少量参数（适配层、低秩分解等），而冻结主干权重，以显著降低计算和存储成本。

## 数学形式
以 LoRA 为例：
$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$，只训练 $A$ 和 $B$。

## 核心要点
1. 冻结预训练权重，只训练少量参数
2. 主要方法：LoRA（低秩分解）、Adapter（插入小模块）、Prefix Tuning（前缀 token）、Prompt Tuning
3. 显著减少显存需求和训练时间
4. 多个任务可共享主干权重，仅切换适配层

## 代表工作
- [[LoRA]]: PEFT 最主流实现
- 广泛用于 VLA 微调场景

## 相关概念
- [[LoRA]]
- [[SFT]]
