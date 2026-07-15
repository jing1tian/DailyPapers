---
type: concept
aliases: [VQ, 向量量化, VQ-VAE, VQ Codebook, 码本量化]
---

# Vector Quantization (VQ)

## 定义

**向量量化**（Vector Quantization, VQ）是一种将连续向量映射到**有限离散码本**（codebook）中最近邻条目的技术，广泛用于将连续特征空间离散化为可学习的离散表征。

## 数学形式

给定连续编码 $z \in \mathbb{R}^d$ 和码本 $\mathcal{C} = \{e_k\}_{k=1}^K$：

$$
z_q = \arg\min_{e_k \in \mathcal{C}} \|z - e_k\|_2
$$

训练时使用直通梯度估计（straight-through estimator）：

$$
\mathcal{L}_\text{VQ} = \|sg[z] - e\|_2^2 + \beta \|z - sg[e]\|_2^2
$$

其中 $sg[\cdot]$ 为停止梯度操作，$\beta$ 为 commitment loss 权重。

## 核心要点

1. **离散化**: 将连续潜在表征映射到有限状态集，便于自回归建模（类比语言 token）
2. **码本学习**: 码本条目通过 EMA 更新或梯度下降与模型协同学习
3. **直通梯度**: 量化操作不可微，通过直通估计将梯度从解码器传回编码器
4. **多模态应用**: 既可量化视觉特征（$Z_t^V$），也可量化动作表征（$Z_t^A$），统一离散化框架

## 代表工作

- [[Lumo-2]]: 同时对视觉潜在（$Z_t^V$）和动作潜在（$Z_t^A$）使用 VQ Codebook，8 个语义动作组

## 相关概念

- [[Latent World Dynamics]]: VQ 将连续世界动态量化为离散潜在码
- [[Latent-Action]]: 动作的潜在表征，通常结合 VQ 进行离散化
- [[Block-wise Autoregression]]: BAR 在量化后的离散 action token 上运行
