---
type: concept
aliases: [LucidDreamer]
---

# LucidDreamer

## 定义
基于深度条件扩散模型的 3D 一致性视频/场景生成方法，用深度 warp + inpainting pipeline 在多视角间保持几何一致性。

## 核心要点
1. 每帧用深度估计 + point cloud warp 预测下一帧的可见区域
2. 遮挡/新出现区域用扩散模型 inpainting 填补
3. 支持从单张图片生成多视角一致的场景漫游视频

## 代表工作
- [[LucidDreamer]]: "LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes" (arXiv 2023)

## 相关概念
- [[Latent Spatial Memory]]
- [[FlashWorld]]
- [[WonderWorld]]
- [[ViewCrafter]]
