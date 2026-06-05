---
type: concept
category: Models
aliases: [Cosmos-Predict2.5, Cosmos Predict2]
tags: [video-generation, world-model]
created: 2026-05-09
---

# Cosmos-Predict2

NVIDIA Cosmos 世界模型系列中的 image-to-video 预测模型，用于生成高质量机器人交互视频。Cosmos-Predict2.5 是其升级版本，提供 2B 参数的 Diffusion Transformer 架构，使用 [[Rectified Flow]] 训练目标。

## 代表工作

- NVIDIA Cosmos
- [[RLDX-1]]: 合成数据流水线用 Cosmos-Predict2 把单帧场景增强结果展开为视频，再由 IDM 标注动作。
- [[OSCAR]]: 以 Cosmos-Predict2.5-2B 为基础模型，通过骨架条件微调实现跨本体动作条件视频生成。
