---
type: concept
aliases: [Voyager, WonderVoyager]
---

# Voyager

## 定义

Voyager 是一种基于 **RGB 点云** 的视频世界模型，用于相机控制的场景漫游视频生成，是 Mirage 的主要对比基线之一。

## 核心要点

1. **RGB 点云条件化**: 将 RGB 彩色点云渲染到目标视角后重编码条件
2. **WorldScore 得分**: 静态场景（77.62）突出，但动态（54.53）和综合（66.08）偏低
3. **RealEstate10K**: PSNR 17.79 / SSIM 0.636 / LPIPS 0.297，整体落后于 Mirage

## 与 Mirage 的对比

| 指标 | Voyager | Mirage |
|------|---------|--------|
| WorldScore Avg | 66.08 | **70.36** |
| Static | **77.62** | 73.60 |
| Dynamic | 54.53 | **67.11** |
| RE10K SSIM | 0.636 | **0.779** |
| Closed-loop SSIM | 0.540 | **0.825** |

## 代表工作

- [[Mirage]]: 在动态一致性和效率上全面超越 Voyager

## 相关概念

- [[RGB 点云记忆]]
- [[Latent Spatial Memory]]
- [[视频扩散条件化]]
