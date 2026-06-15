---
type: concept
aliases: [VAE, 变分自编码器]
---

# Variational Autoencoder

## 定义
一种生成模型，通过编码器将输入映射到潜变量的概率分布（而非确定性向量），利用重参数化技巧实现端到端训练，解码器从采样的潜变量重建输入。

## 数学形式

**ELBO 目标**:
$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\!\left(q_\phi(z|x) \,\|\, p(z)\right)
$$

其中 $q_\phi$ 为编码器，$p_\theta$ 为解码器，$p(z) = \mathcal{N}(0, I)$ 为先验。

## 核心要点
1. 编码器输出均值 $\mu$ 和方差 $\sigma^2$，通过重参数化技巧采样潜变量
2. KL 散度项约束潜变量空间接近标准正态分布
3. 潜空间的连续性和插值性使其适合学习结构化表示
4. 在机器人操作中可用于学习运动先验或技能表示

## 代表工作
- [[HiMem-WAM]]: Stage I 低层 tokenizer 用 VAE 学习运动潜变量

## 相关概念
- [[Reparameterization Trick]]
- [[Hierarchical Latent Action]]
- [[DPFlow]]
