---
type: concept
aliases: [动作掩码, 动作Token掩蔽]
---

# Action Masking

## 定义

在 VLA 训练时，以一定概率随机将数值动作 token 替换为 `<mask>` token 并排除其梯度计算，强迫模型从视觉观测和语言指令中提取更鲁棒的特征，以提升 OOD 泛化能力的数据增强技术。

## 数学形式

$$
\tilde{a}^{(i)} = \begin{cases} \texttt{<mask>} & \text{with probability } p_{\text{mask}} \\ a^{(i)} & \text{otherwise} \end{cases}
$$

掩码 token 不参与损失计算：

$$
\mathcal{L} = -\sum_{i: \tilde{a}^{(i)} \neq \texttt{<mask>}} \log \pi_\theta(a^{(i)} | \tilde{a}^{(<i)}, d, o, l)
$$

## 核心要点

1. 掩码率通常设为 0.4（40% 的动作 token 被遮蔽）
2. 对大模型（4B）有正向收益（+3.2 pts LIBERO 均值），但对小模型（0.8B/2B）略有负面影响（−1.7 to −3.9 pts）
3. 在 OOD 鲁棒性（LIBERO-PRO）上对所有规模均有收益
4. 机制类似 Masked Language Modeling，但应用于动作空间

## 代表工作

- [[CLAP]]: 将 Action Masking 作为可选增强模块，发现规模依赖性

## 相关概念

- [[Language-Action Grounding]]
- [[VLA（视觉-语言-动作模型）]]
- [[掩码 Token 预测]]
