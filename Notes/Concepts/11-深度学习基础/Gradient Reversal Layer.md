---
type: concept
aliases: [梯度反转层, GRL, Gradient Reversal, 对抗解纠缠]
---

# Gradient Reversal Layer（梯度反转层）

## 定义

一种无参数层，前向传播时为恒等映射，反向传播时将梯度乘以 $-\lambda$ 取反，从而使紧接其后的分类器反向训练，强制特征表示无法区分指定属性。

## 数学形式

$$
\text{GRL}(x) = x \quad \text{（前向）}
$$

$$
\frac{\partial \mathcal{L}}{\partial x}\bigg|_{\text{GRL}} = -\lambda \cdot \frac{\partial \mathcal{L}}{\partial \text{GRL}(x)} \quad \text{（反向）}
$$

在对抗解耦训练中，$\lambda > 0$ 通常随训练进度逐渐增大。

## 核心要点

1. **无参数对抗**：不需要额外判别器，通过梯度反转在单一网络内实现对抗训练
2. **域适应起源**：最初用于 Domain-Adversarial Neural Networks（DANN），使特征对域标签不可分
3. **解纠缠应用**：在多模态潜变量解耦中，强制 $z_A$ 不含 $z_B$ 的信息（反之亦然）
4. **与 WGAN 区别**：GRL 是软约束（梯度级别），不保证完全解耦

## 代表工作

- [[DECOWAM]]: 用 GRL 实现底座潜变量 $z_{\text{base}}$ 和手臂潜变量 $z_{\text{arm}}$ 的对抗解纠缠
- Ganin & Lempitsky (2015) "Unsupervised Domain Adaptation by Backpropagation"：GRL 原始论文

## 相关概念

- [[知识蒸馏|Knowledge Distillation]]
- [[底座-手臂因子化|Base-Arm Factorization]]
