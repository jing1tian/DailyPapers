---
type: concept
aliases: [离散瓶颈, 离散表示瓶颈, Discrete Representation Bottleneck, 符号瓶颈]
---

# Discrete Bottleneck（离散瓶颈）

## 定义

离散瓶颈是神经网络中将连续潜变量压缩为离散符号（codebook 索引）的模块，强制模型以有限符号集表示世界状态。常用于世界模型、量化压缩和符号推理。

## 数学形式

**通用形式**：

$$
s = \arg\min_k \| z - e_k \|, \quad e_k \in \mathcal{C} = \{e_1, \ldots, e_K\}
$$

**Gumbel-Softmax 可微近似**（训练时）：

$$
\hat{z} = \text{softmax}\!\left(\frac{(\log \pi + g)}{\tau}\right), \quad g \sim \text{Gumbel}(0,1)
$$

**WP-WM 冻结正交投影**（无可学习 codebook）：

$$
s_t = \arg\max_k (W z_t), \quad W \in \mathbb{R}^{64 \times 32}, \; \nabla W = 0
$$

## 核心要点

1. **可微性问题**：argmax 不可微，常用 Gumbel-Softmax 或 STE（Straight-Through Estimator）近似
2. **语言梯度困境**：语言损失通过瓶颈反向传播会导致[[Symbol Collapse|符号坍缩]]或语义学习失败（WP-WM P3 实验）
3. **写保护方案**：冻结随机投影 + `z.detach()`，彻底隔离语言梯度
4. **与 VQ-VAE 区别**：[[VQ-VAE]] 用可学习 codebook + EMA 更新；WP-WM 用冻结随机正交矩阵，避免 codebook 漂移

## 代表工作

- [[VQ-VAE]]: 可学习 codebook 的离散瓶颈，EMA 更新
- [[GumbelBottleneck]]: Gumbel-Softmax 可微离散化
- [[WP-WM]]: 冻结正交投影 + 写保护，解决语言接地中的结构性失败

## 相关概念

- [[Gumbel-Softmax]] / [[GumbelBottleneck]]
- [[Symbol Collapse]]
- [[Symbol Grounding]]
- [[Stop-Gradient]]
- [[VQ-VAE]]
