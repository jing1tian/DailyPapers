---
type: concept
aliases: [Truncated BPTT, Truncated Backpropagation Through Time, 截断反向传播, TBPTT]
---

# 截断 BPTT（Truncated Backpropagation Through Time）

## 定义

截断 BPTT 是标准 BPTT 的近似变体，在计算序列模型（RNN、展开优化过程）的梯度时只沿时间轴向后展开固定 $K$ 步，以牺牲梯度精度换取内存效率和计算速度。

## 数学形式

完整 BPTT 的梯度：

$$
\frac{\partial L_T}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial L_T}{\partial h_t} \cdot \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \cdot \frac{\partial h_t}{\partial \theta}
$$

截断 BPTT（展开 $K$ 步）的近似梯度：

$$
\frac{\partial L_T}{\partial \theta} \approx \sum_{t=T-K}^{T} \frac{\partial L_T}{\partial h_t} \cdot \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \cdot \frac{\partial h_t}{\partial \theta}
$$

截断偏差以 $\mathcal{O}(\rho^K)$ 速率衰减（$\rho < 1$ 时），见 [[DRILL]] Theorem 1 的 $3B^2\rho^{2K}$ 项。

## 核心要点

1. **截断深度 $K$**: 决定梯度精度与计算代价的权衡；$K=1$ 等价于一阶近似（如 FOMAML）
2. **内存代价**: 完整 BPTT 需存储 $T$ 步的激活值，截断 BPTT 只需 $K$ 步
3. **偏差衰减**: 当展开过程的 Jacobian 谱半径 $\rho < 1$，截断偏差以 $\rho^K$ 指数衰减
4. **在双层优化中的应用**: 用于近似元梯度，$K=2$ 通常已达到 >0.7 余弦相似度（DRILL 实验验证）

## 代表工作

- [[DRILL]]: 使用 $K=2$ 截断 BPTT 估计世界模型训练的元梯度，实验验证 $K=2$ 已超 0.7 余弦相似度阈值

## 相关概念

- [[元梯度]]
- [[双层优化]]
- [[MAML]]
