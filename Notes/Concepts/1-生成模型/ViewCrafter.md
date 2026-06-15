---
type: concept
aliases: [ViewCrafter]
---

# ViewCrafter

## 定义
基于视频扩散模型的新视角合成方法，利用点云 warping 提供几何引导，生成与输入图像视角一致的新视角视频。

## 核心要点
1. 输入：参考图像 + 目标相机轨迹
2. 用深度估计构建点云，将参考图像 warp 到目标视角作为扩散模型的几何条件
3. 视频扩散模型填补遮挡区域并确保时序连贯
4. 主要评测：新视角合成、场景漫游生成

## 代表工作
- [[ViewCrafter]]: "ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis" (arXiv 2024)

## 相关概念
- [[Latent Spatial Memory]]
- [[VideoCrafter2]]
- [[LucidDreamer]]
