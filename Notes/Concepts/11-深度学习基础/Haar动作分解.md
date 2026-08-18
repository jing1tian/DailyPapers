---
type: concept
aliases: [Haar Transform Action Decomposition, Haar Subspace Generation, Scaffold-Residual Decomposition, Haar小波动作分解]
---

# Haar 动作分解

## 定义

将机器人动作块 $A \in \mathbb{R}^{H \times d_a}$ 通过一级正交 Haar 变换分解为 scaffold（逐对均值）和 residual（逐对差值）两组互补系数，精确可逆，无信息损失。

## 数学形式

一级 Haar 变换（对偶步长 $i = 0, \ldots, H/2 - 1$）：

$$
c_i = \frac{a_{2i} + a_{2i+1}}{\sqrt{2}}, \quad d_i = \frac{a_{2i} - a_{2i+1}}{\sqrt{2}}
$$

矩阵形式（$W_H$ 为正交矩阵）：

$$
\begin{bmatrix} C \\ D \end{bmatrix} = W_H A, \quad A = W_H^T \begin{bmatrix} C \\ D \end{bmatrix}
$$

精确重建（无重建误差）：

$$
a_{2i} = \frac{c_i + d_i}{\sqrt{2}}, \quad a_{2i+1} = \frac{c_i - d_i}{\sqrt{2}}
$$

## 核心要点

1. **Scaffold $C$**：编码每对动作步的均值（逐对运动水平），捕获块级运动语义。
2. **Residual $D$**：编码每对动作步的差值（对内变化），恢复局部精细动作，精确补充 scaffold。
3. **正交性保证**：$\|W_H\| = 1$，能量守恒，无参数，固定不可学习。
4. **有序生成**：先生成 scaffold $\hat{C}$，再以 $\text{sg}(\hat{C})$ 为条件生成 residual $\hat{D}$，两阶段共享专家网络但用不同嵌入和掩码区分；stop-gradient 防止残差梯度覆盖 scaffold。
5. **Handoff 适配接口**：两阶段均接收 handoff 条件 $g_{r,k}$，在不改变行为语义的前提下适配续接上下文。

## 代表工作

- [[BICPO-VLA]]：将 Haar 子空间生成引入 VLA 动作块生成，与行为条件化动作纤维结合解决异步接管问题

## 相关概念

- [[行为条件化动作纤维]]
- [[Action Chunking]]
- [[Flow Matching]]
- [[Handoff状态滚动]]
