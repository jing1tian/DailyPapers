---
type: concept
aliases: [Free Bits, Per-dimension Free Bits, 自由比特正则化, 按维度KL下界]
---

# Free-Bits Regularization

## 定义

VAE 训练中防止后验坍塌（posterior collapse）的一种 KL 正则化技术：对每个隐变量维度 $d$ 单独设置 KL 散度的最小下界 $C$，若某维度的 KL 已低于 $C$，则不再对其施加梯度压力，允许该维度自由编码信息。

## 数学形式

标准 VAE KL 项替换为：

$$
\mathcal{L}_{KL} = \sum_{d=1}^{D} \max\Bigl(C,\ D_{\mathrm{KL}}\bigl(q_\phi(z_d | \mathbf{x}) \| \mathcal{N}(0,1)\bigr)\Bigr)
$$

其中：
- $D$: 隐变量总维度
- $C > 0$: free-bits 阈值（单位：nat 或 bit），FM-VLA 中 $C = 1.0\ \text{nat}$
- $z_d$: 第 $d$ 维隐变量

## 核心要点

1. **后验坍塌问题**: 标准 VAE 的 KL 正则化有时过强，导致编码器输出先验 $\mathcal{N}(0,1)$（后验坍塌），隐变量完全不携带信息
2. **Free-Bits 机制**: 当 $D_{\text{KL}}(z_d) < C$ 时，$\max$ 的梯度为零，解码器被迫从该维度提取信息
3. **Per-dimension 粒度**: 每个维度独立计算，允许部分维度充分利用、部分维度保持近先验——比全局 $\beta$-VAE 更细粒度
4. **在 FM-VLA 中的应用**: 与 [[Masked ELBO]] 结合训练 Force-VAE，确保 768 个隐维度（$K \times 96$）都有效编码 wrench 历史信息

## 代表工作

- Kingma et al. (2016), "Improving Variational Inference with Inverse Autoregressive Flow"（首次提出 free bits 概念）
- [[FM-VLA]]: 用 per-dimension free-bits（$C=1.0$）训练 wrench 历史的 Force-VAE

## 相关概念

- [[VAE]]
- [[Masked ELBO]]
- [[Perceiver-IO]]
