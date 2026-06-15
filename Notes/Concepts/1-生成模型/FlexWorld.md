---
type: concept
aliases: [FlexWorld]
---

# FlexWorld

## 定义

FlexWorld 是一种相机可控的视频世界模型，专注于灵活的相机轨迹生成，是 Mirage 的对比基线之一。

## 核心要点

1. **相机控制**: 支持用户定义相机轨迹的视频生成
2. **WorldScore 得分**: 综合 60.23，相机控制得分 84.43（高于 Mirage 的 55.36）
3. **闭环一致性弱**: RE10K 闭环 PSNR_C 仅 12.20，SSIM_C 仅 0.428，漂移明显

## 与 Mirage 的对比

| 指标 | FlexWorld | Mirage |
|------|-----------|--------|
| WorldScore Avg | 60.23 | **70.36** |
| Camera Ctrl | **84.43** | 55.36 |
| RE10K PSNR_C | 12.20 | **20.05** |
| RE10K SSIM_C | 0.428 | **0.825** |

## 代表工作

- [[Mirage]]: 在持久空间一致性（闭环指标）上大幅超越 FlexWorld

## 相关概念

- [[视频扩散条件化]]
- [[Latent Spatial Memory]]
- [[条件视频生成]]
