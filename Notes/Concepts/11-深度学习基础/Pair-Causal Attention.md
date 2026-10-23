---
type: concept
aliases: [pair-causal attention, PC attention, Pair-Causal 注意力]
---

# Pair-Causal Attention

## 定义

Pair-Causal Attention 是一种专为运动-力矩配对 token 序列设计的因果注意力掩码机制：在联合去噪时，保证运动 token $A_i^+$ 不能 attend 同配对的未来力矩 token $W_i^+$，但 $W_i^+$ 可以 attend $A_i^+$，所有 token 均可 attend 早期配对。

## 数学形式

对未来序列 $[A_1^+, W_1^+, A_2^+, W_2^+, \ldots, A_H^+, W_H^+]$，注意力掩码为下三角形，但额外屏蔽 $A_i^+$ 对 $W_i^+$ 的 attention：

$$
\text{Mask}[A_i^+, W_i^+] = -\infty \quad (\text{运动不看同对未来力矩})
$$

$$
\text{Mask}[W_i^+, A_i^+] = 0 \quad (\text{力矩可看同对运动})
$$

## 核心要点

1. **物理动机**: 运动命令先于接触力产生，因果关系为 $A_i^+ \to W_i^+$，而非反向。
2. **与全双向注意力的区别**: CW-WAM 消融实验表明，使用全双向注意力（CW-WAM w/o PC）会导致位置误差增加 2.96 mm（95% CI [2.05, 3.89]）。
3. **联合去噪**: 所有配对 token 在同一扩散步骤内联合去噪，非自回归，但通过掩码施加因果约束。

## 代表工作

- [[CW-WAM]]: 提出 Pair-Causal Attention，用于运动-力矩联合预测中的结构化因果建模

## 相关概念

- [[世界动作模型（WAM）]]
- [[Flow-Matching]]
- [[长程力矩记忆（LTM）]]
