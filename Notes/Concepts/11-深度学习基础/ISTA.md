---
type: concept
aliases: [Iterative Shrinkage-Thresholding Algorithm]
---

# ISTA

## 定义
Iterative Shrinkage-Thresholding Algorithm：求解 L1 正则化稀疏优化问题（Lasso）的迭代算法，等价于对梯度步骤施加软阈值（soft-thresholding）操作。

## 数学形式
$$z^{(t+1)} = \mathcal{S}_\lambda(z^{(t)} - \alpha \Phi^T(\Phi z^{(t)} - x))$$
其中 $\mathcal{S}_\lambda$ 为软阈值算子。

## 核心要点
1. 等价于将稀疏编码（sparse coding）的推理解释为 RNN 展开
2. LISTA（Learned ISTA）将其参数化为神经网络层
3. 在视觉皮层模型中作为 sparse coding inference 的计算等价

## 代表工作
- [[Toward a mechanistic understanding of inference in visual cortex and diffusion models]]: 将 V1 的 ISTA 推理等价为 diffusion model

## 相关概念
- [[Stable Diffusion]]
