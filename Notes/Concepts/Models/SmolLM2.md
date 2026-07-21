---
type: concept
aliases: [SmolLM2, SmolLM 2]
---

# SmolLM2

## 定义

SmolLM2 是 Hugging Face 发布的小型高效语言模型系列，作为 [[SmolVLM2]] 的语言解码器骨干，专为资源受限场景下的语言理解与生成设计。

## 核心要点

1. **极致轻量**: 参数规模从 135M 到 1.7B，目标是在移动端/边缘端部署。
2. **社区友好**: 完全开源，基于 HuggingFace 生态，Transformers 库原生支持。
3. **在 VLA 中的角色**: 作为 [[SmolVLM2]] 的语言解码器，与 [[SigLIP]] 视觉编码器协同处理多模态输入，最终为机器人策略提供语义条件。
4. **训练数据**: 在大规模高质量语料上训练，指令遵循能力强。

## 代表工作

- [[SmolVLA]]: 使用 SmolLM2 作为 VLM 骨干的语言解码器，实现低成本机器人策略
- [[SmolVLM2]]: SmolLM2 + SigLIP 构成的多模态视觉语言模型

## 相关概念

- [[SmolVLM2]]: 以 SmolLM2 为解码器的多模态模型
- [[SigLIP]]: 与 SmolLM2 配对的视觉编码器
- [[Vision-Language-Action Model]]: SmolLM2 作为语言骨干服务的下游任务
