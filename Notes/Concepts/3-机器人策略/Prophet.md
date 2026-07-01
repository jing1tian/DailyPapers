---
type: concept
aliases: [Prophet World Model]
---

# Prophet

## 定义

一种面向机器人操作的世界模型方法，用于动作条件视觉预测，在 LIBERO 等 benchmark 上作为世界模型生成质量的竞争基线。

## 核心要点

1. **定位**: 动作条件世界模型，关注视觉预测质量（PSNR/SSIM）
2. **性能**: LIBERO PSNR 26.12，SSIM 0.8887（被 A2World-sim 以 26.64 / 0.8957 超越）
3. **与 A2World 对比**: 缺乏姿态引导历史采样和多具身形态大规模预训练

## 代表工作

- [[A2World]]: 将 Prophet 作为世界模型生成质量的对比基线

## 相关概念

- [[Action-Conditioned World Model]]
- [[Action Conditioning]]
- [[LIBERO]]
