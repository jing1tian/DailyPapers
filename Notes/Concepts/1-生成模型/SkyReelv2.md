---
type: concept
aliases: [SkyReel v2, SkyReel-v2]
---

# SkyReelv2

## 定义
快手开源的视频生成模型，基于扩散 Transformer 架构，支持高分辨率长视频生成，在开源社区中被广泛用作视频生成基线。

## 数学形式

标准视频扩散目标（flow matching）：
$$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\left[\|v_\theta(x_t, t, c) - (x_0 - \epsilon)\|^2\right]$$

## 核心要点
1. 快手技术团队出品，完全开源权重和推理代码
2. 使用 Flow Matching 训练框架，生成质量较高
3. SANA-Video 等后续工作将其作为基线对比
4. 支持图像引导的视频生成

## 代表工作
- [[SANA-Video]]: 与 SkyReelv2 进行了质量和速度对比

## 相关概念
- [[DiT]]
- [[MovieGen]]
- [[MAGI-1]]
