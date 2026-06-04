---
type: concept
aliases: [CoT, Chain of Thought, 链式思维, 思维链]
---

# Chain-of-Thought Reasoning

## 定义

Chain-of-Thought（CoT）推理是指让语言模型在给出最终答案之前，生成一系列中间推理步骤，从而提升复杂任务的解题准确率。

## 数学形式

标准 CoT 生成过程：

$$
p(a \mid x) = \sum_{c} p(a \mid c, x) \cdot p(c \mid x)
$$

其中 $c$ 为中间推理链（reasoning chain），$x$ 为输入，$a$ 为最终输出。

## 核心要点

1. **显式中间步骤**: 将复杂推理分解为可观测的语言步骤，提升解题可解释性
2. **Few-shot CoT**: 在提示中提供带推理链的示例，引导模型模仿推理过程
3. **Zero-shot CoT**: "Let's think step by step" 即可激活推理能力
4. **具身 CoT（Embodied CoT）**: 将 CoT 推理扩展到机器人操作，推理链包含任务理解、空间定位、子目标规划和动作描述
5. **推理与行动分离**: ERVLA 等工作发现训练时的 CoT 监督比测试时的显式推理链更重要

## 代表工作

- [[ECoT]]: 首次将具身 CoT 引入 VLA，推理链作为动作生成的自回归前缀
- [[ERVLA]]: 通过 Reasoning Dropout 使模型训练时内化 CoT、推理时无需显式生成
- [[ThinkAct]]: 利用 CoT 进行扩散策略条件化

## 相关概念

- [[VLA]]
- [[VLM]]
- [[Embodied Forcing]]
- [[Flow Matching]]
