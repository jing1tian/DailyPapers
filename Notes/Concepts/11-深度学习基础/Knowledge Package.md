---
type: concept
aliases: [知识包, 知识包 ICL, Knowledge Context]
---

# Knowledge Package（知识包）

## 定义

In-Context Learning 中预先构建的冻结上下文结构，包含任务指导、运动参考、视觉参考、可执行 Helper、检查与恢复逻辑及元数据，注入模型推理上下文以在不更新权重的前提下注入任务知识。

## 核心要点

1. 在测试前一次性构建，推理时权重不更新（冻结）
2. 包含 6 类内容：任务指导 / 运动参考 / 视觉参考 / 可执行 Helper / 检查与恢复 / 元数据
3. 知识包质量直接决定 ICL 效果上限，跨任务迁移性较弱

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: 在 12 个具身 benchmark 中引入知识包，结合闭环 System 1/2 协作实现 54.17% SR

## 相关概念

- [[In-Context Learning]]
- [[System 2 Thinking]]
