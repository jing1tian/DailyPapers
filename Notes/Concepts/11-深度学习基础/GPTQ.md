---
type: concept
aliases: [Generative Pre-trained Transformer Quantization, 二阶权重量化]
---

# GPTQ

## 定义

基于二阶（Hessian）近似的大语言模型权重量化方法，逐层、逐列地最小化量化误差，在 W4A16 精度下能将 LLM 参数压缩至接近原始精度。

## 数学形式

对每一列权重 $w_q$ 量化后，用 Hessian 逆矩阵（$H^{-1}$）更新剩余列，以补偿量化误差：

$$
\delta W_{:,q} = -\frac{w_q - \operatorname{quant}(w_q)}{[H^{-1}]_{qq}} \cdot [H^{-1}]_{:,q}
$$

其中 $H = 2 X X^T$ 为当前层的 Hessian 矩阵，$X$ 为输入激活。

## 核心要点

1. **逐列量化**: 按列顺序量化，量化一列后立即用 Hessian 更新剩余列，误差累积最小
2. **无需激活量化**: 仅量化权重（W4），激活保持高精度
3. **校准数据**: 只需少量（~128 条）校准样本估计 Hessian 矩阵
4. **适合 LLM 侧**: DiT 侧经旋转后分布更均匀，RTN（四舍五入）即可，不必用 GPTQ

## 代表工作

- [[Omega-QVLA]]: 将 GPTQ 用于 VLA 语言骨干的 W4 权重量化（配合 SVD-Hadamard 旋转）

## 相关概念

- [[后训练量化（PTQ）]]
- [[Hadamard 矩阵]]
- [[SVD]]
