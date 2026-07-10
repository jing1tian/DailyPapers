---
type: concept
aliases: [LongCat Video, 长视频生成模型]
---

# LongCat-Video

## 定义
面向长视频生成的视频扩散基础模型（Xunliang Cai et al., 2025），在 1080p 长时程视频生成质量上具有竞争力，引入 HPSv3 作为视觉质量奖励基础。

## 核心要点
1. 专注长视频时序一致性建模
2. 使用 HPSv3 进行视觉质量评估，为后续奖励设计提供参考（被 [[LingBot-Video]] 采用）
3. 在 RBench 上开源模型中排名较低（0.437），被 LingBot-Video 等后续模型超越

## 代表工作
- LongCat-Video (Cai et al., arXiv 2510.22200)

## 相关概念
- [[Video Diffusion Model]]
- [[Video Foundation Model]]
- [[DiT]]
