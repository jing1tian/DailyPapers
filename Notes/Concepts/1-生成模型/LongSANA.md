---
type: concept
aliases: [LongSANA, Long Video SANA]
---

# LongSANA

## 定义
SANA-Video 中用于生成分钟级长视频的策略，基于 autoregressive chunk-by-chunk 生成，配合 Block Linear Diffusion Transformer 实现高效长序列视频合成。

## 核心要点
1. 将长视频分解为多个 chunk，逐块自回归生成
2. 利用 linear attention 替代标准 attention 降低长序列计算复杂度
3. 与 Self-Forcing 策略配合，减少 teacher forcing 引起的 distribution shift
4. 支持 720×1280 分辨率、分钟级时长的视频生成

## 代表工作
- [[SANA-Video]]: LongSANA 的提出论文，NVIDIA 出品的高效视频生成模型

## 相关概念
- [[Self-Forcing]]
- [[MAGI-1]]
- [[DiT]]
