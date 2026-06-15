---
type: concept
aliases: [频率调和采样, 温度加权采样, Frequency-Tempered Batch Weighting]
---

# Frequency-Tempered Sampling

## 定义

一种多数据集混合训练中的采样权重策略，通过对各数据集大小取温度 $T$ 调和的幂次来平衡大小数据集的采样频率，避免大数据集完全主导训练。

## 数学形式

$$w_i \propto n_{\text{frames},i}^{1/T}$$

其中 $T > 1$ 为温度参数，$n_{\text{frames},i}$ 为第 $i$ 个数据集的总帧数。

## 核心要点

1. $T=1$ 退化为按帧数比例采样（大数据集完全主导）
2. $T \to \infty$ 退化为均匀采样（每个数据集等权）
3. $T=3$ 是 OSCAR 使用的值，在数据集大小差异悬殊时（最小 96 条 vs 最大 78,273 条）提供合理平衡
4. 概念上与语言模型多语言训练中的温度采样相似（如 mC4 采样策略）

## 代表工作

- [[OSCAR]]: 使用 $T=3$ 平衡 7 个大小差异极大的机器人和人类数据集

## 相关概念

- [[Diffusion Transformer]]
