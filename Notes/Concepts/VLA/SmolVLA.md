---
type: concept
aliases: [Smol VLA]
---

# SmolVLA

## 定义

SmolVLA 是 HuggingFace 提出的小型 [[VLA|视觉-语言-动作模型]]，设计目标是在低计算资源下实现实用的机器人操作能力，强调模型轻量化与易部署性。

## 核心要点

1. 参数规模较小（"Smol" 意为小型），适合边缘部署
2. 基于预训练视觉语言模型微调，具备一定的 [[Out-of-Distribution Generalization|OOD 泛化]] 能力
3. 在 [[VLA-REPLICA]] 评测中 ID 平均成功率 26%，OOD 30%（优于多数从头训练的 IL 基线）

## 代表工作

- [[VLA-REPLICA]]: 作为被评测的 VLA 模型之一

## 相关概念

- [[VLA]]
- [[ACT]]
- [[π0]]
- [[模仿学习]]
