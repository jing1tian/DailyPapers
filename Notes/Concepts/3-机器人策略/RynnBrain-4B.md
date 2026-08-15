---
type: concept
aliases: [RynnBrain 4B, RynnBrain]
---

# RynnBrain-4B

## 定义

一个 4B 参数规模的冻结视觉-语言模型，在 Temporal GRPO 中用于从任务指令和初始观测生成候选语义任务阶段序列。

## 核心要点

1. **用途**: 给定任务指令 $l$ 和初始观测 $o_0$，输出任务的候选语义阶段列表，供 Stage Compiler 进一步编译
2. **使用方式**: 冻结推断，不参与 RL 训练，仅在 Stage Compiler 阶段调用
3. **规模**: 4B 参数，属于中等规模 VLM
4. **局限**: 对指令语义模糊的任务，生成的阶段质量可能下降

## 代表工作

- [[TemporalGRPO]]: 使用 RynnBrain-4B 作为阶段生成模块

## 相关概念

- [[VLA（视觉-语言-动作模型）]]
- [[阶段条件优势]]
- [[时序信用分配]]
