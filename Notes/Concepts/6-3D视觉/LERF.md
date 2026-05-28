---
type: concept
aliases: [LERF, Language Embedded Radiance Fields]
---

# LERF

## 定义

LERF 是一种将 CLIP 语言嵌入集成到 NeRF 中的方法，使得 3D 场景的每个位置都具有语义语言特征，支持通过自然语言 query 在 3D 场景中定位和分割物体。

## 核心要点

1. **语言-3D 融合**: 在 NeRF 训练时同时优化几何密度和 CLIP 语义特征场
2. **多尺度嵌入**: 在不同尺度提取 CLIP 特征并融合，处理不同粒度的 query
3. **零样本定位**: 无需额外标注，通过文本 query 直接定位场景中的物体
4. **局限**: 基于 NeRF（慢），在 3DGS 场景中被 LangSplat 等方法取代

## 代表工作

- Kerr et al. 2023 (ICLR 2024): LERF 原始论文

## 相关概念

- [[LangSplat]]
- [[3D Gaussian Splatting]]
- [[NeRF]]
- [[CLIP]]
- [[OpenGaussian]]
- [[ReferSplat]]
