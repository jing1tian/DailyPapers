---
type: concept
aliases: [Video DiT, Video Diffusion Transformer]
---

# Video Diffusion Transformer

## 定义

Video Diffusion Transformer（Video DiT）是将 Diffusion Model 的去噪过程与 Transformer 架构结合，专门用于视频生成的模型，以 [[DiT]] 为骨干网络，支持时空注意力建模。

## 核心要点

1. **时空注意力**: 在空间维度和时间维度同时建模帧间相关性
2. **潜在空间去噪**: 通常在压缩的潜在空间（VAE Latent）中操作，降低计算成本
3. **相机条件控制**: 可通过 Plücker 射线或相机嵌入实现精确相机轨迹控制
4. **双向感受野**: 与自回归模型不同，可利用所有帧的上下文信息

## 代表工作

- [[HYWorld2]]: WorldStereo 2.0 以 Video DiT 为主干，配合 GGM 和 SSM++ 记忆机制实现多视图一致性世界扩展

## 相关概念

- [[DiT]]
- [[Diffusion Model]]
- [[3D Gaussian Splatting]]
- [[Distribution Matching Distillation]]
