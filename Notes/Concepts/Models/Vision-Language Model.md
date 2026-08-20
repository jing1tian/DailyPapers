---
type: concept
aliases: [VLM, Vision-Language Model, 视觉语言模型, 多模态大模型]
---

# Vision-Language Model（视觉语言模型，VLM）

## 定义

能够同时理解和处理**视觉信息（图像/视频）与语言信息（文本）**的大规模预训练模型，通过跨模态对齐实现视觉问答、图像描述、视觉推理等任务。

## 核心要点

1. **典型架构**: 视觉编码器（ViT/CLIP）+ 语言模型（LLM）+ 跨模态投影层（MLP/Q-Former）。常见代表：LLaVA、Qwen-VL、InternVL、PaliGemma。
2. **在 VLA 中的角色**: [[Vision-Language-Action Model|VLA]] 模型通常以预训练 VLM 为 Backbone，在其上追加动作头（Action Head），利用 VLM 的视觉-语言理解能力支持语言条件机器人控制。
3. **Qwen-VL 系列**: [[LoopVLA]] 使用 Qwen3 VLM 作为 Backbone（Qwen3OFT 和 Qwen3FM 变体），参数量约 2.2B。
4. **能力边界**: VLM 擅长高层语义理解，但直接用于精确操控时可能损失低层几何线索（[[LoopVLA]] 发现的问题）。

## 代表工作

- LLaVA、Qwen-VL、InternVL: 通用 VLM 代表工作。
- [[Pi0|π₀]]、[[LoopVLA]]、[[OpenVLA]]: 基于 VLM Backbone 构建的 VLA 模型。

## 相关概念

- [[Vision-Language-Action Model]]
- [[多模态表示]]
- [[Transformer]]
- [[循环块]]
