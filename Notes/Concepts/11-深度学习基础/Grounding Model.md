---
type: concept
aliases: [视觉定位模型, Visual Grounding, Open-Vocabulary Detector]
---

# Grounding Model

## 定义

给定自然语言描述，在图像中定位（输出边界框或 mask）对应对象的视觉语言模型，代表如 Grounding DINO、GDINO-1.5 等。

## 核心要点

1. **开放词汇**：不依赖固定类别集合，支持任意文本查询
2. **与 VLM 协同**：RoboScript 中与 Set-of-Mark 和 VLM 三方投票确定对象身份
3. **应用场景**：机器人操作中的对象识别、指代消解

## 代表工作

- Grounding DINO: 结合 DINO 检测器与 BERT 文本编码的开放词汇检测
- [[RoboScript]]: 使用 grounding model 参与对象身份三方投票

## 相关概念

- [[VLM]]
- [[In-Context Learning]]
