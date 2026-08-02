---
type: concept
aliases: [GigaBrain]
---

# GigaBrain

## 定义
GigaWorld-1 中用于视频扩散世界模型的核心神经网络骨干，基于 DiT（Diffusion Transformer）架构，融合了 SageAttention 等效率优化，用于机器人策略的世界模型评估。

## 核心要点
1. 作为 GigaWorld-1 系统的生成骨干
2. 集成 SageAttention 加速长序列视频帧的注意力计算
3. 支持 LoRA 适配下游机器人任务
4. 通过 AdamW 优化器 + RoPE 位置编码训练

## 代表工作
- [[GigaWorld1]]: GigaBrain 是其核心生成模型组件，负责视频扩散预测

## 相关概念
- [[DiT]]
- [[SageAttention]]
- [[World Model]]
