---
type: concept
aliases: [表示锚定, VL Anchoring, Representation Anchoring, 视觉语言锚定]
---

# Vision-Language Anchoring

## 定义

VLA 微调时维护一个冻结的预训练 VLM 副本作为 anchor（teacher），通过对所有 decoder 层施加层级 Frobenius 范数蒸馏损失，防止行为克隆微调导致[[灾难性遗忘]]视觉语义能力。

## 数学形式

单层锚定损失：

$$
\mathcal{L}_{anchor}^{(u)} = \left\|\mathbf{H}_u^S[\mathbf{m}] - \mathbf{H}_u^A[\mathbf{m}]\right\|_F^2
$$

总锚定损失（对所有 decoder 层求平均）：

$$
\mathcal{L}_{anchor} = \frac{1}{|\mathcal{D}|} \sum_{u \in \mathcal{D}} \mathcal{L}_{anchor}^{(u)}
$$

其中 $\mathbf{H}_u^S$ 为第 $u$ 层 student（训练中 VLA）的隐状态，$\mathbf{H}_u^A$ 为 anchor（冻结 VLM）的隐状态，$\mathbf{m}$ 为视觉和文本 token 位置集合。

## 核心要点

1. **全层级蒸馏**: 对所有 transformer decoder 层施加约束，而非仅最后一层
2. **选择性对齐**: 仅对视觉和文本 token（$\mathbf{m}$）计算蒸馏损失，不包括动作 token，避免干扰动作预测
3. **零额外数据**: 完全复用演示数据，无需额外视觉语言数据集
4. **实验效果**: 可将 GQA 视觉推理准确率保留至 70%（标准 BC 在 10K 步内损失 94%）

## 代表工作

- [[AnchorAlign]]: 提出 Vision-Language Anchoring 的核心论文，与 Language-Action Alignment 联合使用

## 相关概念

- [[知识蒸馏]]
- [[灾难性遗忘]]
- [[Language-Action Alignment]]
- [[行为克隆]]
- [[VLA（视觉-语言-动作模型）]]
