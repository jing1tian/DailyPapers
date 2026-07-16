---
type: concept
aliases: [OpenAI Sora, Sora视频生成]
---

# Sora

## 定义

OpenAI 于 2024 年推出的大规模文本生成视频模型，基于 Diffusion Transformer（DiT）架构，能生成高分辨率、长时序的连贯视频，是视频生成领域的里程碑工作。

## 核心要点

1. 以 DiT 替代传统 U-Net，在 token 化的视频 patch（spacetime patch）上做扩散
2. 以物理一致性和时序连贯性著称，但追求像素级写实导致监督信号冗余
3. 世界模型视角：Sora 隐式学习了部分物理规律，但非显式状态建模

## 代表工作

- Sora 本身是代表性视频生成工作
- [[Orca]]: 将 Sora 类的像素生成范式与自身的隐状态预测范式进行对比

## 相关概念

- [[DiT]]
- [[Flow Matching]]
- [[Stable Diffusion 3.5]]
- [[Next-State Prediction]]
