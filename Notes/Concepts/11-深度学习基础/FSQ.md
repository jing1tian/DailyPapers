---
type: concept
aliases: [Finite Scalar Quantization]
---

# FSQ

## 定义
有限标量量化，用于替代 VQ-VAE 中的 codebook 查询，将连续向量的每个维度独立量化到有限个离散值，无需维护 codebook，避免 codebook 崩溃问题。

## 数学形式
$$z_q = \text{round}(\tanh(z) \cdot L/2) \cdot 2/L, \quad L \in \{5, 7, 8, ...\}$$

## 核心要点
1. 每个特征维度独立量化为 $L$ 个等间距值（如 $L=8$ 时有 8 个值）
2. 无需 codebook，也无需 commitment loss 或 straight-through estimator
3. codebook size = $\prod_i L_i$，各维度 $L_i$ 可独立设置
4. 在 SONIC 中用于运动序列的 token 化，替代 VQ 做离散表示

## 代表工作
- [[FSQ]]：Mentzer et al. 2023，Finite Scalar Quantization
- [[SONIC]]：人形机器人运动跟踪，用 FSQ 做运动 token 化

## 相关概念
- [[VQ-VAE]]
- [[Diffusion Model]]
