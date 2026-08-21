---
type: concept
aliases: [SVD, Stable Video Diffusion, Stability AI Video]
---

# Stable Video Diffusion

## 定义
Stability AI 发布的开源**视频扩散模型**，基于图像扩散模型（Stable Diffusion）扩展至时序建模，以高分辨率图像为条件生成短视频片段，被广泛用作机器人世界模型的骨干网络。

## 数学形式

SVD 以 EDM（Elucidated Diffusion Model）框架训练，去噪目标为：

$$
\mathcal{L} = \mathbb{E}_{x, \varepsilon, \sigma}\left[\lambda(\sigma)\left\|\hat{x}_\theta(x + \sigma\varepsilon;\, \sigma) - x\right\|_2^2\right]
$$

其中 $\sigma$ 为噪声水平，$\lambda(\sigma)$ 为损失权重，$\hat{x}_\theta$ 为 U-Net 去噪器。

## 核心要点
1. **图生视频**: 以单帧图像为条件，自回归生成 14~25 帧视频
2. **U-Net 时空注意力**: 在 2D U-Net 基础上插入时序注意力层
3. **帧间一致性**: 通过跨帧注意力保持时序连贯
4. **下游适配友好**: LoRA 微调 + ControlNet 扩展被广泛验证有效

## 代表工作
- [[Mask2Real-WM]]: 以 SVD 为骨干，分解为掩码动力学（WM1）与 RGB 渲染（WM2）两阶段
- [[Ctrl-World]]: 基于 SVD 构建并行夹爪操作世界模型

## 相关概念
- [[Video Diffusion Model]]
- [[ControlNet]]
- [[LoRA]]
- [[Classifier-Free Guidance]]
