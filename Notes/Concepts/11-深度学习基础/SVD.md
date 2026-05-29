---
type: concept
aliases: [Singular Value Decomposition, 奇异值分解, SVD]
---

# SVD

## 定义
奇异值分解（Singular Value Decomposition）：将矩阵 $A$ 分解为 $A = U\Sigma V^T$ 的经典线性代数方法，广泛用于压缩、降维和低秩近似。

## 数学形式
$$A = U\Sigma V^T, \quad A \in \mathbb{R}^{m \times n}$$

低秩近似：
$$A \approx U_k \Sigma_k V_k^T, \quad k \ll \min(m, n)$$

## 核心要点
1. $U$、$V$ 为正交矩阵，$\Sigma$ 为对角奇异值矩阵
2. 截断 SVD 可做最优低秩近似（Eckart-Young 定理）
3. 在模型压缩中：用低秩 SVD 近似大权重矩阵以减少参数量
4. Ω-QVLA 用 SVD 分解压缩 [[DiT]] action head 的权重矩阵

## 代表工作
- [[Ω-QVLA]]：利用 SVD 低秩分解压缩 VLA 的 DiT action head

## 相关概念
- [[DiT]]
- [[LoRA]]
- [[DoRA]]
