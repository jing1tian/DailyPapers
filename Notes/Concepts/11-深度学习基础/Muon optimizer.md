---
type: concept
aliases: [Muon, Muon Optimizer, 动量正交化优化器]
---

# Muon Optimizer

## 定义
一种基于梯度动量正交化（Momentum Orthogonalization）的优化器，通过对梯度应用 Nesterov 动量后再进行正交化处理，使参数更新方向更加正交，加速收敛并提升大规模模型训练稳定性。

## 数学形式

核心思想：对动量梯度 $m_t$ 应用正交化变换（如 Newton-Schulz 迭代）：

$$
\tilde{m}_t = \text{Orthogonalize}(m_t)
$$

更新规则：$\theta_{t+1} = \theta_t - \eta \cdot \tilde{m}_t$

## 核心要点
1. **正交化梯度**：使不同参数维度的更新方向相互正交，减少梯度干扰
2. **高学习率兼容**：允许使用较高的峰值学习率（如 $1\times10^{-2}$），加速收敛
3. **大模型适配**：在十亿参数级模型（1.3B、5B）上表现稳定

## 代表工作
- [[RepWAM]]: 使用 Muon optimizer 训练 1.3B 和 5B WAM 模型，峰值学习率 $1\times10^{-2}$

## 相关概念
- [[AdamW]]
- [[条件流匹配]]
- [[扩散变换器]]
