---
type: concept
aliases: [Efficient Vision Transformer, Efficient ViT]
---

# EViT

## 定义
一种基于 attention score 融合非重要 token 的 ViT 加速方法——把低 attention 的 token 融合（fuse）成少量 "inattentive" token 而非直接丢弃，保留信息的同时降低序列长度。

## 核心要点
1. 识别低注意力的 token，将其融合为一个或多个 summary token
2. 比 DynamicViT 更保守：不丢弃信息，而是压缩
3. 对下游任务精度损失更小

## 代表工作
- Liang et al., Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations, ICLR 2022

## 相关概念
- [[DynamicViT]]
- [[APS-VLA]]
