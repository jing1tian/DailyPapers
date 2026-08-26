---
type: concept
aliases: [SoftVQ, soft vector quantization, 软向量量化]
---

# SoftVQ

## 定义
向量量化的软化版本，允许 latent 向量以连续权重分配到多个码本条目，而非硬性分配到最近邻，减少训练梯度问题并保留更细粒度的信息。

## 数学形式
$$z_q = \sum_k w_k e_k, \quad w_k = \text{softmax}(-\|z - e_k\|_2^2 / \tau)$$

温度 $\tau \to 0$ 退化为硬 VQ，$\tau \to \infty$ 退化为均匀平均。

## 核心要点
1. 避免 VQ 的直通梯度估计问题
2. 在 latent dynamics 建模中用于离散化动作/状态表示
3. 与 VQVAE 相比梯度更稳定，codebook 利用率更高

## 代表工作
- [[LD4WAM]]: 用 SoftVQ 离散化 latent dynamics 表示

## 相关概念
- [[Vector Quantization]]
- [[VQVAE]]
- [[Latent Dynamics Model]]
