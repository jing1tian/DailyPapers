---
type: concept
aliases: [LingBot Video, embodied MoE video pretraining, 具身智能专用视频预训练]
---

# LingBot-Video

## 定义
专门为具身智能设计的 DiT-based 视频预训练框架，用 Mixture-of-Experts 替代 dense Transformer，降低通用视频生成模型在物理真实性和计算效率上的 domain mismatch。

## 核心要点
1. 基于 DiT 架构，用 MoE 层替换 dense FFN，按场景类型和物理属性路由计算
2. 避免通用视频模型"优化视觉保真度"导致的负迁移，专注物理一致性
3. 与 LingBot-World-Infinity 同属一个研究体系（LingBot 系列）
4. 下游支持：操作策略、世界模型等具身任务的预训练骨干

## 代表工作
- Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence (2607.07675)

## 相关概念
- [[DiT]]
- [[MoE]]
- [[World-Infinity]]
- [[AlayaWorld]]
