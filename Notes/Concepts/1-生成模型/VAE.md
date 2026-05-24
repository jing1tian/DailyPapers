---
type: concept
aliases: [VAE, Variational Autoencoder, 变分自编码器]
---

# VAE（Variational Autoencoder）

## 定义
通过将输入编码为潜空间分布（而非确定性向量）并从该分布采样重建的生成模型，实现了结构化的潜表示学习。

## 数学形式

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

## 核心要点
1. **编码器** $q_\phi(z|x)$：输出潜分布的均值 $\mu$ 和方差 $\sigma^2$。
2. **重参数化技巧**：$z = \mu + \sigma \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0,I)$，使梯度可反传。
3. **解码器** $p_\theta(x|z)$：从潜向量重建输入。
4. KL 散度正则项使潜空间趋向标准正态分布，实现连续采样。

## 代表工作
- Kingma & Welling（2013）：VAE 原始论文
- [[潜扩散模型]]：使用 VAE 编解码器作为像素-潜空间桥梁
- [[OrbiSim]]: OrbiSim-Vision 使用 VAE 在潜空间执行扩散去噪

## 相关概念
- [[潜扩散模型]]
- [[Diffusion Model]]
