---
type: concept
aliases: [System 2, 慢速思维, Slow Thinking, 高层规划器]
---

# System 2 Thinking（慢速/分析思维）

## 定义

借鉴认知科学双过程理论，在具身 AI 中指慢速、分析式的高层规划模块（如大型视觉语言模型），负责任务理解、状态推理、子目标分解和恢复策略制定。

## 核心要点

1. 慢速、高推理成本，适合复杂语义理解和多步规划
2. 通常由大型视觉语言模型（如 GPT-6-Astra）担任
3. 通过闭环反馈审查 [[System 1 Thinking]] 的执行结果并纠错

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: GPT-6-Astra 作为 System 2，每 4 步审查 π0.5 执行结果，闭环协作大幅提升成功率

## 相关概念

- [[System 1 Thinking]]
- [[In-Context Learning]]
- [[Knowledge Package|知识包]]
