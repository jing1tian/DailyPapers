---
type: concept
aliases: [Vector Quantized Behavior Transformer, VQ-BeT]
---

# VQ-BeT

## 定义
基于向量量化的行为 Transformer，将连续动作空间离散化后用 autoregressive Transformer 建模，解决多模态动作分布问题。

## 数学形式
$$a_t = \text{Decode}(z_t), \quad z_t \sim \text{VQ-Codebook}$$

## 核心要点
1. 用 VQ-VAE 将连续动作离散化为 codebook token
2. Transformer 自回归预测动作 token 序列
3. 相比 BC Transformer 更好处理多模态（一个场景有多种正确动作）
4. 在 Push-T、Kitchen 等 benchmark 上表现强劲

## 代表工作
- [[VQ-BeT 原论文]]：Lee et al., 2024，提出方法

## 相关概念
- [[Diffusion Policy]]
- [[ACT]]
- [[VQ-VAE]]
- [[Action Chunking]]
