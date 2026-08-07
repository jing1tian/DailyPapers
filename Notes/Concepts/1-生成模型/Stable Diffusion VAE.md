---
type: concept
aliases: [SD VAE, LDM VAE, Latent Diffusion VAE]
---

# Stable Diffusion VAE

## 定义

[[Stable Diffusion]] 中用于将图像编码到低维潜空间（并解码回像素空间）的变分自编码器，是隐扩散模型（LDM）的核心组件之一，提供约 8× 或 16× 的空间压缩比。

## 核心要点

1. **压缩比**: 典型配置将 $H \times W \times 3$ 图像压缩至 $\frac{H}{8} \times \frac{W}{8} \times C$ 的潜变量（$C$ 通常为 4 或 16 通道）
2. **冻结使用**: 在下游任务中通常**冻结** SD VAE 编码器，仅用于提取潜变量特征，不参与更新
3. **Mind-VLA 中的用途**: 将目标物体三视图（顶/正/侧三张 RGB）离线编码为堆叠潜变量 $\mathbf{Z}_{m(\ell)}$，压缩比约 48×；模型的目标物体查询预测该潜变量

## 代表工作

- [[Mind-VLA]]: 冻结 SD VAE 将目标物体三视图离线编码为训练监督目标
- [[Stable Diffusion]]: 原始提出工作，LDM 框架

## 相关概念

- [[VAE]]
- [[Stable Diffusion]]
- [[Diffusion Policy]]
