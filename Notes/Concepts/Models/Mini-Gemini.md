---
type: concept
aliases: [Mini-Gemini, MiniGemini]
---

# Mini-Gemini

## 定义
轻量级多模态大语言模型，采用双视觉 encoder 架构（低分辨率语义 encoder + 高分辨率细节 encoder），在保持小参数量的同时提升视觉理解精度。

## 核心要点
1. 双 encoder：低分辨率 [[CLIP]] 做语义理解，高分辨率 ConvNeXt 保留细节
2. Mining tokens：高分辨率 encoder 的特征引导 LLM attention 聚焦到重要区域
3. 常用于比较 MLLMs 的 benchmark（MME、SEED、DocVQA 等）

## 代表工作
- [[Mini-Gemini]]: "Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models" (CVPR 2024)

## 相关概念
- [[CLIP]]
- [[InternVL]]
- [[Eagle]]
- [[EVA-02]]
