---
type: concept
aliases: [Low-Rank Adaptation, 低秩适配, LoRA微调]
---

# LoRA

## 定义

LoRA（Low-Rank Adaptation）是一种参数高效微调方法，通过在预训练权重旁插入低秩分解矩阵，以极少额外参数实现模型适配，同时冻结原始权重。

## 数学形式

对预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 学习低秩增量：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$。

## 核心要点

1. **参数高效**: 仅学习 $B, A$ 两个小矩阵，可训练参数减少至原来的 0.1%–1%。
2. **原始权重冻结**: $W_0$ 保持不变，适配通过低秩增量叠加实现。
3. **推理合并**: 训练完成后可将 $\Delta W$ 合并回 $W_0$，推理无额外开销。
4. **局限**: 对生成策略进行 LoRA 微调时，权重更新仍会破坏行为先验（见 [[FlowDAgger]] Table 3）。

## 代表工作

- [[FlowDAgger]]: LoRA-DAgger 作为 baseline，先验保留仅剩 0.30（损失 −0.66）

## 相关概念

- [[知识蒸馏]]
- [[灾难性遗忘]]
- [[Noise Policy]]
