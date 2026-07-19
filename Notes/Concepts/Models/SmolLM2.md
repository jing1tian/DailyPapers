---
type: concept
aliases: [SmolLM2, SmolLM-2]
---

# SmolLM2

## 定义

SmolLM2 是 Hugging Face 发布的一系列轻量级语言模型（135M/360M/1.7B 参数），针对低资源设备和高效推理优化，是 [[SmolVLM2|SmolVLM-2]] 的语言解码器骨干。

## 核心要点

1. **规格**: 提供 135M、360M、1.7B 三个参数规模，适合不同资源约束场景
2. **架构**: 基于 Transformer 解码器，采用分组查询注意力（GQA）和旋转位置编码（RoPE）
3. **训练数据**: 在大规模多语言文本数据上预训练，覆盖代码、数学和通用语料
4. **VLA 应用**: 在 [[SmolVLA]] 中作为 [[SmolVLM2|SmolVLM-2]] 的语言解码器，负责处理语言指令和生成机器人动作所需的高层语义特征

## 代表工作

- [[SmolVLA]]: 使用 SmolLM2 作为 VLM 骨干的语言部分，结合 [[SigLIP]] 视觉编码器构成 SmolVLM-2，实现轻量级 VLA 模型

## 相关概念

- [[SmolVLM2|SmolVLM-2]]: 以 SmolLM2 为语言骨干的多模态模型
- [[SigLIP]]: SmolVLM-2 中配合 SmolLM2 使用的视觉编码器
- [[SmolVLM]]: SmolLM2 的前代多模态集成版本
