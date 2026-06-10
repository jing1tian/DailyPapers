---
type: concept
aliases: [SAM3, Segment Anything with Concepts]
---

# SAM3

## 定义

Segment Anything with Concepts 的缩写，在 SAM（Segment Anything Model）基础上融合概念语义理解，支持基于文本概念描述的精确实例分割，常与视觉语言模型配合使用完成语义驱动的掩码生成。

## 核心要点

1. **语义驱动分割**: 接受文本概念描述作为提示，生成对应语义区域的掩码
2. **与 VLM 协同**: 通常搭配 Qwen3-VL 等视觉语言模型完成"识别→分割"两阶段流程
3. **精细掩码质量**: 继承 SAM 的高质量边界，适合用于动态物体过滤等需要精确区域的任务

## 代表工作

- [[Mirage]]: 用 Qwen3-VL-2B + SAM3 生成天空与动态物体掩码，防止瞬态内容污染持久记忆

## 相关概念

- [[SAM]]
- [[SAM-2]]
- [[Qwen3-VL]]
