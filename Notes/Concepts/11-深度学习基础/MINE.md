---
type: concept
aliases: [MINE, Mutual Information Neural Estimation, 互信息神经网络估计]
---

# MINE

## 定义

MINE（Mutual Information Neural Estimation）是一种用神经网络估计高维随机变量之间互信息的方法，基于 Donsker-Varadhan 表示将 KL 散度（互信息）转化为对偶形式进行估计。

## 数学形式

基于 DV 表示：

$$I(X;Z) \geq \mathbb{E}_{p(x,z)}[T_\theta(x,z)] - \log \mathbb{E}_{p(x)p(z)}[e^{T_\theta(x,z)}]$$

用神经网络 $T_\theta$ 近似统计量，最大化下界来估计互信息。

## 核心要点

1. **高维适用**: 传统直方图法无法处理高维空间，MINE 可以
2. **估计方差大**: 在高维输入（如视觉特征）下估计方差通常较大，结论需谨慎解读
3. **VLA 理论**: ITBound-VLA 用 MINE 估计 VLA 能力与鲁棒性之间的互信息 tradeoff

## 代表工作

- Belghazi et al. 2018: MINE 原始论文, ICML 2018

## 相关概念

- [[InfoNCE]]
- [[因果推断]]
