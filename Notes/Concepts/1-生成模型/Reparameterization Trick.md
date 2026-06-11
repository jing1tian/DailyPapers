---
type: concept
aliases: [重参数化技巧, 重参数化, Reparametrization Trick]
---

# Reparameterization Trick

## 定义
变分自编码器（VAE）训练中的关键技术，通过将随机采样过程分解为确定性变换加独立噪声的形式，使梯度可以反传过采样操作。

## 数学形式

$$
z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

等价于 $z \sim \mathcal{N}(\mu, \sigma^2 I)$，但梯度对 $\mu$ 和 $\sigma$ 可微。

## 核心要点
1. 将不可微的采样操作转化为可微的确定性变换
2. $\varepsilon$ 与网络参数无关，梯度只流经 $\mu$ 和 $\sigma$
3. 是 VAE 能够端到端训练的关键
4. 适用于任何可表示为位置-尺度族的分布

## 代表工作
- [[HiMem-WAM]]: 低层 tokenizer 中用于采样运动潜变量 $z^l_t$
- [[Variational Autoencoder]]: 该技术的原始应用场景

## 相关概念
- [[Variational Autoencoder]]
- [[Hierarchical Latent Action]]
