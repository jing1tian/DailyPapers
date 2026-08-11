---
type: concept
aliases: [DriveLaW, Driving Language and World model]
---

# DriveLaW

## 定义
将语言模型与 video world model 结合用于端到端自动驾驶的框架，在推理时依赖未来帧生成来辅助 action 预测。

## 核心要点
1. 以 VLM 理解场景语义，以 video diffusion model 预测未来帧
2. 推理时必须运行 video generation branch，延迟高
3. 被 [[SimWAM]] 的"训练时用、推理时扔"策略超越

## 相关概念
- [[SimWAM]]
- [[DriveWAM]]
- [[WAM]]
