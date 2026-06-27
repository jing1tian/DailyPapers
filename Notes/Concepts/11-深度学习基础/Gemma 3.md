---
type: concept
aliases: [Gemma 3, Gemma3]
---

# Gemma 3

## 定义

Gemma 3 是 Google 发布的开源轻量级大语言模型（LLM）/视觉语言模型（VLM）系列，参数规模覆盖从小到中等，常作为 VLA 模型的语言/视觉-语言骨干网络（backbone）。

## 核心要点

1. **开源轻量**: 相比闭源大模型，Gemma 3 提供可本地部署、可微调的开源权重，适合机器人 VLA 系统的端到端训练与推理优化（如 torch.compile、CUDA graphs）
2. **VLM 骨干角色**: 搭配视觉编码器（如 [[SigLIP]]）构成视觉语言模型，为 VLA 的动作头提供语言指令理解和视觉语境表征
3. **KV-Cache 复用**: 在 VLA 架构中，Gemma 3 的中间层 KV 常通过类似 π0.5 的连接器（connector）被动作专家通过 Cross-Attention 读取，避免重复计算

## 代表工作

- [[ABC]]: ABC-VLA 以 Gemma 3 + SigLIP 构成的 VLM 作为骨干，设计了 Cross Attention、Cross Attention + FAST、AdaLN 三种连接器变体，将动作头与冻结/微调的 VLM 表征对接

## 相关概念

- [[SigLIP]]
- [[VLA（视觉-语言-动作模型）]]
- [[Cross-Attention]]
- [[FAST]]
- [[Pi05|π0.5]]
