---
type: concept
aliases: [ZR0, ZR-Zero]
---

# ZR-0

## 定义

基于 [[GR00T N1]] 架构改进的视觉-语言-动作基础模型，提供 ZR-0base 预训练权重，作为多个后续 VLA 工作（包括 VLA-Δ）的初始化起点。

## 核心要点

1. **架构基础**：基于 GR00T N1 设计，使用 [[Diffusion Transformer]] 动作专家
2. **强基线性能**：在 LIBERO 平均成功率 97.8%（Spatial 97.4% / Object 99.4% / Goal 97.8% / Long 96.4%）
3. **RoboTwin 表现**：Clean 88.70% / Random 87.98%，Clean-to-Random 下降 0.72%
4. **作为初始化**：VLA-Δ、Motus 等工作使用 ZR-0base 权重初始化

## 代表工作

- [[VLA-Delta]]: 以 ZR-0base 为初始化权重，在 LIBERO 和 RoboTwin 上超越 ZR-0

## 相关概念

- [[GR00T N1]]
- [[Diffusion Transformer]]
- [[Flow Matching]]
- [[pi0.5]]
- [[VLA（视觉-语言-动作模型）]]
