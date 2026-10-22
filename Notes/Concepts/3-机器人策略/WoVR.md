---
type: concept
aliases: [WoVR, World-of-Video-Robotics]
---

# WoVR

## 定义

WoVR（World-of-Video Robotics）是一种基于视频生成的机器人世界模型方法，作为机器人视觉预测领域的基线模型之一，用于对比接触密集型操作任务上的 RGB 预测质量。

## 核心要点

1. 基于视频生成框架建模机器人操作场景
2. 在 PhysCanvas 的 Table 2 对比中，外观变化强（appearance shift strong），场景一致性不如 PhysCanvas
3. 定性对比中会改变袋口等场景细节

## 代表工作

- [[PhysCanvas]]: 将 WoVR 作为 RGB 预测质量对比基线（PSNR、SSIM、LPIPS、FID、FVD、EEF 误差等指标均低于 PhysCanvas）

## 相关概念

- [[Video World Model]]
- [[Visuo-Tactile World Models]]
