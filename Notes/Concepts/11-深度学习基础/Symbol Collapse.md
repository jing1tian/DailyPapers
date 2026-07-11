---
type: concept
aliases: [符号坍缩, 模式坍缩, Codebook Collapse, 符号多样性丢失]
---

# Symbol Collapse（符号坍缩）

## 定义

符号坍缩是离散瓶颈系统中的失效模式：模型训练后仅使用词表中极少数符号（或只用 1-2 个），导致符号多样性丧失、瓶颈失去区分能力。类似于 GAN 的 mode collapse 但发生在离散符号空间。

## 数学形式

设符号词表大小为 $K$，有效符号数为：

$$
N_{\text{active}} = |\{s : P(s) > \epsilon\}|, \quad s \in \{0, \ldots, K-1\}
$$

坍缩条件：$N_{\text{active}} \ll K$（例如 WP-WM 中：2.2 / 64，仅用 ~3%）

## 核心要点

1. **触发条件**：语言梯度通过[[GumbelBottleneck|Gumbel-Softmax 瓶颈]]反向传播时，强力拉拽某几个符号使其独占输出
2. **抗坍缩策略均失败**：高温、低学习率、谱归一化、熵奖励均能保持多样性但使语义学习失败（≤9.2%）——这是**结构性权衡**而非优化问题
3. **根本原因**：梯度优化目标（语义准确）与符号均匀使用（多样性）相互冲突
4. **解决方案**：[[Stop-Gradient|梯度写保护]] + 冻结投影，完全切断语言梯度对符号层的修改

## 代表工作

- [[WP-WM]]: 系统证明 Gumbel-Softmax 瓶颈在语言梯度下不可避免符号坍缩，提出写保护修复

## 相关概念

- [[Discrete Bottleneck]]
- [[GumbelBottleneck]]
- [[Gumbel-Softmax]]
- [[Stop-Gradient]]
- [[Symbol Grounding]]
