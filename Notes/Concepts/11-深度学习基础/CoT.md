---
type: concept
aliases: [Chain of Thought, 链式思维, 思维链]
---

# CoT

## 定义
在 LLM 推理时通过中间步骤（"let's think step by step"）显式生成推理过程，提升复杂问题的准确率。

## 数学形式
$$P(y|x) = \sum_{c} P(y|x, c) \cdot P(c|x)$$

其中 $c$ 是中间推理链（chain），通过增大 $P(c|x)$ 中有效推理路径的概率来提升最终答案质量。

## 核心要点
1. Few-shot CoT：在 prompt 中给出带推理链的示例
2. Zero-shot CoT："Let's think step by step" 即可触发
3. Self-consistency：多次采样 CoT 路径后投票，进一步提升准确率

## 代表工作
- Wei et al., 2022：提出 CoT prompting
- [[DA-CoT]]：动作感知 CoT for VLA

## 相关概念
- [[ReAct]]
- [[SFT]]
