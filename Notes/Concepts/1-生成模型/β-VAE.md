---
type: concept
aliases: [Beta-VAE, beta-VAE, β VAE, 贝塔变分自编码器]
---

# β-VAE

## 定义

β-VAE 是在标准 [[VAE]] 基础上引入权重超参数 $\beta$ 的变分自编码器，通过增大 KL 散度项的权重来约束隐空间的紧凑性和解耦性。

## 数学形式

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - \beta \, D_{\mathrm{KL}}\!\left(q_\phi(z|x) \,\|\, p(z)\right)
$$

其中 $\beta > 1$ 时比标准 VAE 施加更强的隐空间正则化；$\beta=1$ 退化为标准 VAE。

## 核心要点

1. **标准 VAE 的局限**：$\beta=1$ 时正则化不足，隐变量容易出现纠缠（entangled）表示
2. **$\beta>1$ 的效果**：强迫 encoder 学习更稀疏、解耦的隐表示，但过大会牺牲重建质量
3. **在隐动作模型中的应用**：将动作编码为紧凑隐向量，KL 项防止 encoder 将未来帧外观信息直接压缩进隐动作（防止 appearance leakage）

## 代表工作

- [[LAWM-3D]]: 用 β-VAE 框架学习 3D 感知隐动作，重建目标为 RGB + 深度图联合重建

## 相关概念

- [[VAE]]
- [[Latent-Action]]
- [[LAM]]
