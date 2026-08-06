---
type: concept
aliases: [Language-Image Value, Language-conditioned Image Value]
---

# LIV

## 定义
Language-Image Value：用语言-图像对齐表示学习训练的价值函数，用于估计当前视觉状态相对于语言目标的完成程度。

## 核心要点
1. 利用 VLM 预训练的语言-视觉对齐能力
2. 输出状态值估计，指导 reward shaping 或 failure detection
3. 无需人工定义 reward，通过语言目标自动计算

## 代表工作
- [[ValueFormer]]: 对比 LIV 作为 value learning baseline

## 相关概念
- [[HIL]]
- [[CLIP]]
