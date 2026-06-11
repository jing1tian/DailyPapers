---
type: concept
aliases: [重参数化技巧, 重参数化, Reparameterization]
---

# Reparameterization Trick

## 定义

将随机变量 $z \sim \mathcal{N}(\mu, \sigma^2)$ 的采样过程重写为 $z = \mu + \sigma \odot \varepsilon,\ \varepsilon \sim \mathcal{N}(0, I)$，从而将随机性从计算图中分离，使梯度能够通过采样操作反向传播。

## 数学形式

$$
z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

## 核心要点

1. 解决 VAE 中采样操作不可微的问题
2. 将不可微的采样 $z \sim q_\phi(z|x)$ 变为对确定性参数 $\mu, \sigma$ 的可微计算
3. 是 VAE 及其变体能够端到端训练的关键

## 代表工作

- [[VAE]]: 最早在 VAE 中提出该技巧用于变分推断
- [[HiMem-WAM]]: 在低层潜在动作 Tokenizer 中使用重参数化采样

## 相关概念

- [[变分自编码器]]
- [[VAE]]
