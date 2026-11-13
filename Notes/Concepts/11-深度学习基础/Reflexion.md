---
type: concept
aliases: [反思框架]
---

# Reflexion

## 定义

LLM agent 框架，在任务失败后将自然语言反思（reflection）写入 context，使 agent 在不更新权重的情况下从错误中学习，常用于代码生成和推理任务。

## 核心要点

1. **语言反思而非梯度**：失败后生成文字描述的原因分析，存入 episodic memory
2. **短期 vs 长期记忆**：短期 context + 长期 memory 的双层结构
3. **局限**：通常在同一 episode 内学习，不跨 episode 蒸馏经验
4. **与 WICL 的区别**：WICL 跨 episode 蒸馏经验到 Task Memory，并有世界提供实际物理反馈

## 代表工作

- Shinn et al. (2023): 原始 Reflexion 论文
- [[RoboScript]]: 将类似反思机制扩展到多轮、多 episode 的机器人操作

## 相关概念

- [[In-Context Learning]]
- [[Coding Agent]]
- [[Recursive Self-Improvement]]
