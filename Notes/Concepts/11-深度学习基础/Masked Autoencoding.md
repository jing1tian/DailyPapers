---
type: concept
aliases: [MAE, Masked Autoencoders, 掩码自编码器]
---

# Masked Autoencoding

## 定义

**Masked Autoencoding**（MAE）是一种自监督预训练方法：随机掩盖输入的一部分（图像 patch、视频帧、token），训练模型从可见部分重建被掩盖内容。

## 数学形式

$$
\mathcal{L}_{MAE} = \mathbb{E}_{\mathbf{x}, \mathcal{M}} \left[ \| \mathbf{x}_{\mathcal{M}} - f_\theta(\mathbf{x}_{\bar{\mathcal{M}}}) \|^2 \right]
$$

**符号说明**:
- $\mathbf{x}$: 输入（图像/视频/序列）
- $\mathcal{M}$: 被掩码的子集索引
- $\bar{\mathcal{M}}$: 可见部分索引
- $f_\theta$: 编码器-解码器网络

## 核心要点

1. 掩码率通常较高（75% 图像 patch），强迫模型学习语义级特征而非像素统计
2. 与 [[Next-Token Prediction]] 的区别：MAE 是双向上下文、随机掩码；NTP 是单向因果、自回归
3. 在视频世界模型中，掩码时间帧形成**视频预测**的一种特殊形式
4. [[JEPA]] 在潜空间中执行 MAE 风格的预测，避免像素重建

## 代表工作

- [[WorldModelRoadmap]]: MAE 作为世界模型自监督预训练的核心范式
- VideoMAE: 视频领域 MAE 扩展

## 相关概念

- [[JEPA]]
- [[Next-Token Prediction]]
- [[World Model]]
- [[Scaling Law]]
