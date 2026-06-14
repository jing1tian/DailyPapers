---
type: concept
aliases: [FlashWorld, Flash-World]
---

# FlashWorld

## 定义
基于显式 RGB 点云记忆的快速视频世界模型，用点云缓存已见帧的 3D 信息以维持新视角生成时的空间一致性。

## 核心要点
1. 用 RGB 点云存储场景的几何+颜色信息
2. 生成新视角时，将点云 project 到当前相机，作为条件图像输入扩散模型
3. 相比 [[Latent Spatial Memory]]，存储和重投影在 RGB 空间进行，计算成本较高

## 代表工作
- [[FlashWorld]]: "FlashWorld: World Models with Explicit Point Cloud Memory"

## 相关概念
- [[Latent Spatial Memory]]
- [[WonderWorld]]
- [[LucidDreamer]]
- [[ViewCrafter]]
