---
type: concept
aliases: [Hadamard Transform, Hadamard Rotation, 哈达玛矩阵, 哈达玛变换]
---

# Hadamard 矩阵

## 定义

一种元素为 $\pm 1$ 的正方矩阵，满足 $H H^T = nI$（$n$ 为维度）。归一化版本 $H/\sqrt{n}$ 是正交矩阵，具有将向量能量均匀扩散到所有维度的性质。

## 数学形式

归一化 Hadamard 矩阵（$n$ 为 2 的幂次）：

$$
H_n = \frac{1}{\sqrt{n}} \begin{pmatrix} H_{n/2} & H_{n/2} \\ H_{n/2} & -H_{n/2} \end{pmatrix}, \quad H_1 = [1]
$$

关键性质（激活异常值均衡）：

$$
\|z H\|_\infty \leq \frac{\|z\|_2}{\sqrt{n}}
$$

任意向量 $z$ 经 Hadamard 旋转后，无穷范数（最大通道幅值）被均衡到 $\ell_2$ 范数的 $1/\sqrt{n}$。

## 核心要点

1. **能量均衡性**: 将单个通道集中的能量均匀扩散到所有通道，消除激活异常值
2. **计算高效**: 可用快速 Hadamard 变换（FHT）以 $O(n \log n)$ 计算，无需存储矩阵
3. **与 SVD 互补**: SVD 旋转均衡权重通道能量，Hadamard 旋转均衡激活通道能量，两者组合效果更强
4. **数据无关**: 不依赖具体数据分布，无需校准

## 代表工作

- [[Omega-QVLA]]: 复合 SVD-Hadamard 旋转，用于 VLA 模型 W4A4 量化中消除 DiT 激活异常值
- [[QuaRot]]: 利用 Hadamard 旋转进行 LLM 量化
- [[FlatQuant]]: 基于旋转的 LLM 量化

## 相关概念

- [[SVD]]
- [[后训练量化（PTQ）]]
- [[GPTQ]]
