---
type: concept
aliases: [System 1, 快速思维, Fast Thinking, 低层控制器]
---

# System 1 Thinking（快速/直觉思维）

## 定义

借鉴认知科学中的双过程理论，在具身 AI 中指快速、反应式的低层控制模块（如 VLA），负责将高层规划转化为具体动作执行，无需大量推理。

## 核心要点

1. 快速、低延迟，适合精细的电机控制
2. 通常由专用小模型（如 [[π0.5]]）担任
3. 与 [[System 2 Thinking]] 协作：System 2 规划，System 1 执行

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: π0.5 作为 System 1，GPT-6-Astra 作为 System 2，闭环协作实现 54.17% SR（vs 基线 15.56%）

## 相关概念

- [[System 2 Thinking]]
- [[π0.5]]
- [[VLA|Vision-Language-Action Model]]
