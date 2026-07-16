---
type: concept
aliases: [自监督学习, SSL]
---

# Self-Supervised Learning

## 定义

一类无需人工标注的学习范式，通过从数据本身构造监督信号（如遮掩预测、对比学习、下一帧预测）训练模型，是大规模预训练的基础方法论。

## 数学形式

以 Masked Autoencoder 为例：

$$
\mathcal{L}_{\text{SSL}} = \mathbb{E}_{x \sim \mathcal{D}}\left[\|\hat{x}_{\text{masked}} - x_{\text{masked}}\|^2\right]
$$

## 核心要点

1. 监督信号来自数据内部结构（时序、空间、语义一致性）
2. 三大范式：对比学习（SimCLR、MoCo）、遮掩预测（MAE、BERT）、预测编码（JEPA）
3. 大规模 SSL 预训练后微调是当前 CV/NLP Foundation Model 的标准路线

## 代表工作

- [[V-JEPA]]: 在 ViT 隐空间做预测编码的视频 SSL
- [[Orca]]: 将 SSL 视频预训练扩展为通用世界状态预训练

## 相关概念

- [[Unconscious Learning]]
- [[ViT]]
- [[Latent Matching Loss]]
