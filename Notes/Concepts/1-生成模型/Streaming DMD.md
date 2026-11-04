---
type: concept
aliases: [Streaming Distribution Matching Distillation, 流式 DMD]
---

# Streaming DMD（流式分布匹配蒸馏）

## 定义

[[DMD2]]（分布匹配蒸馏）的流式扩展版本，专为自回归 chunk 生成场景设计，在因果世界模型的后训练阶段与 [[Consistency Distillation]] 联合使用，实现少步（1–4 步）高质量视频生成。

## 核心要点

1. **流式设计**：DMD2 原始版本面向独立帧/clip，Streaming DMD 适配 chunk 自回归生成，保持 chunk 间时序一致性
2. **后训练阶段**：在 Stage III 与 [[Consistency Distillation]] 联合蒸馏，兼顾单 chunk 质量（CD）和跨 chunk 分布一致性（DMD）
3. **少步推理**：蒸馏后模型支持 1–4 步 NFE（神经函数评估）生成，配合 [[GLA]] 的流式 KV 节省，大幅提升实时推理能力
4. **EMA 教师网络**：继承 DMD2 的 EMA 教师对齐机制，防止生成质量退化

## 代表工作

- [[XiaomiAutoWM2]]: Streaming DMD 用于 Stage III 后训练，与 [[Consistency Distillation]] 联合蒸馏，支持 XiaomiAutoWM2 的实时流式推理

## 相关概念

- [[DMD2]]: 分布匹配蒸馏的原始方法（前驱工作）
- [[Consistency Distillation]]: Stage III 联合使用的另一蒸馏方法
- [[Flow-Matching]]: Stage I 预训练使用的基础生成目标
- [[GLA]]: 流式注意力，与 Streaming DMD 共同支撑实时推理
