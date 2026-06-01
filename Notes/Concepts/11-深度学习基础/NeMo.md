---
type: concept
aliases: [NVIDIA NeMo, NeMo Framework]
---

# NeMo

## 定义
NVIDIA 开源的大规模语言模型和多模态模型训练框架，提供从预训练到 RLHF 微调的完整流水线，支持 Megatron-Core 张量并行和流水线并行。

## 核心要点
1. 基于 PyTorch Lightning 和 Megatron-Core 构建，支持数千 GPU 规模训练
2. 原生支持 FP8 训练（H100/B100）和序列并行
3. 包含 NeMo-Aligner 组件，支持 PPO、GRPO、DPO 等对齐算法
4. ProRL Agent 使用 NeMo 作为训练后端之一

## 代表工作
- [[ProRL]]: 在 NeMo 上实现 Rollout-as-a-Service

## 相关概念
- [[VeRL]]
- [[SkyRL]]
- [[GRPO]]
