---
type: concept
aliases: [Kahan Summation, Kahan summation algorithm, 补偿求和, Error Compensation]
---

# Kahan 加法

## 定义
Kahan 加法（Kahan Summation）是一种数值算法，通过维护一个"补偿"变量来追踪浮点加法中的累积舍入误差，使最终结果的精度显著优于朴素求和。

## 数学形式

标准 Kahan 求和算法（伪代码）：

$$
\begin{aligned}
&c \leftarrow 0 \quad \text{// 误差补偿项} \\
&\text{for each } x_i: \\
&\quad y \leftarrow x_i - c \\
&\quad t \leftarrow \text{sum} + y \\
&\quad c \leftarrow (t - \text{sum}) - y \quad \text{// 捕获精度损失} \\
&\quad \text{sum} \leftarrow t
\end{aligned}
$$

**效果**：对 $n$ 个数的和，误差从 $O(n\epsilon)$ 降至 $O(\epsilon)$（$\epsilon$ 为机器精度）。

## 核心要点
1. **应用于低精度权重更新**：在 bf16 权重旁维护一个 bf16 残差（补偿）变量，使小梯度能被"积累"
2. **与 fp32 copy 的区别**：Kahan 加法只需额外 1 个 bf16 变量（不是完整的 fp32 副本），内存开销约 50%（相比 fp32 copy 的 100%）
3. **实现复杂度**：需要修改优化器更新步骤，而非只是改变参数存储类型

## 代表工作
- Kahan (1965): 原始算法提出
- [[VLA-BF16-UpdateErasure]]: 将 Kahan summation（bf16 残差）列为内存受限时的替代方案

## 相关概念
- [[BFloat16]]: Kahan 加法主要解决 bf16 更新的累积精度问题
- [[fp32 master copy]]: 精度更好的解决方案，内存充裕时优先
- [[随机舍入]]: 另一种替代方案
