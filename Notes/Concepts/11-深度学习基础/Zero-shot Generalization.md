---
type: concept
aliases: [零样本泛化, Zero-Shot, 零样本迁移]
---

# Zero-shot Generalization（零样本泛化）

## 定义

模型在从未见过的任务、场景或形态上直接完成推理/执行，无需任何领域内训练样本或微调。

## 核心要点

1. 依赖预训练阶段积累的通用知识进行迁移
2. 对具身 AI 而言，意味着机器人可在新环境/新任务中直接工作
3. 与 [[In-Context Learning]] 结合（few-shot 或 zero-shot ICL）

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: 真实世界验证部分使用零样本控制（无领域演示），5 类形态均直接运行

## 相关概念

- [[In-Context Learning]]
- [[Embodied AI|具身智能]]
