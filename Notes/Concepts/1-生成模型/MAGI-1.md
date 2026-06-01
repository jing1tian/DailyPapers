---
type: concept
aliases: [MAGI1, Magi-1]
---

# MAGI-1

## 定义
Sand AI 发布的视频生成模型，以自回归扩散（Autoregressive Diffusion）为核心，支持长视频的因果生成，被 SANA-Video 等工作用作性能对比基线。

## 核心要点
1. 采用 chunk-level 自回归生成策略，逐段生成视频保持时序一致性
2. 支持长视频生成，单次推理可生成分钟级别内容
3. 与 MovieGen、SkyReelv2 同级别的商业/半开源视频生成模型
4. SANA-Video 在速度和质量上与 MAGI-1 进行对比

## 代表工作
- [[SANA-Video]]: 将 MAGI-1 作为对比基线

## 相关概念
- [[DiT]]
- [[SkyReelv2]]
- [[MovieGen]]
- [[CausVid]]
