---
type: concept
category: 1-生成模型
aliases: [Genie Envisioner, Genie2]
tags: [video-generation, world-model, action-conditioned]
created: 2026-06-05
---

# Genie Envisioner

## 定义

Genie Envisioner 是 Google DeepMind Genie 系列的视频预测世界模型，支持动作条件视频生成，可用于机器人策略评估场景。

## 核心要点

1. 属于显式动作条件类视频世界模型；
2. 在机器人操作视频生成任务中作为主流基线；
3. 在 OSCAR 评估基准（200 片段，4 个本体）上：PSNR 23.29、SSIM 0.838、LPIPS 0.140、FVD 15.37。

## 代表工作

- [[OSCAR]]: 在多本体评估基准上以 PSNR 24.24 / FVD 7.08 超越 Genie Envisioner（2B vs ~7B 参数）。

## 相关概念

- [[Diffusion Transformer]]
- [[Rectified Flow]]
- [[Kinema4D]]
