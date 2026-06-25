---
type: concept
aliases: [SD, Latent Diffusion]
---

# Stable Diffusion

## 定义
潜空间扩散模型（Latent Diffusion Model），先用 VAE 将图像压缩到低维潜空间，再在潜空间上做扩散去噪生成，大幅降低扩散模型的计算开销，是文生图领域的代表性开源模型。

## 数学形式
$$\mathcal{L} = \mathbb{E}_{z_0,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2\right]$$

## 核心要点
1. 潜空间扩散大幅降低分辨率带来的计算成本，是后续多数文生图/视频生成模型的基础范式
2. 常作为"组合泛化失败"等扩散模型理论分析工作的标准测试对象
3. 与 [[DDIM]] 等采样器搭配使用以加速推理

## 相关概念
- [[DDIM]]
- [[Qwen-Image]]
