---
type: concept
aliases: [PaliGemma, PaliGemma-3B]
---

# PaliGemma

## 定义
Google DeepMind 发布的轻量级视觉语言模型，将 SigLIP2 视觉编码器与 Gemma 语言模型结合，支持图像理解、OCR、视觉问答等任务，完全开源。

## 核心要点
1. 架构：SigLIP2 ViT（视觉）+ Gemma-2B/3B（语言）
2. 以较小参数量（3B）实现强视觉理解能力
3. 常用作 VLA 的 backbone（因为轻量可微调）
4. PaliGemma-2 进一步提升了图像理解和推理能力

## 代表工作
- Beyer et al. (2024): "PaliGemma: A versatile 3B VLM for transfer"
- [[UAM]]: 用 PaliGemma 研究 VLA 训练中的遗忘问题

## 相关概念
- [[SigLIP2]]
- [[VLM]]
- [[Qwen3-VL]]
