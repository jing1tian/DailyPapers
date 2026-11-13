---
type: concept
aliases: [RSI, 递归自我改进, 递归自我提升]
---

# Recursive Self-Improvement (RSI)

## 定义

agent 在冻结权重下，通过读取自身评估记录、识别失败模式并将 Guidance 写入 context，从而在不更新梯度的情况下持续提升性能的机制。

## 数学形式

$$
G_{j+1} = V_\theta(G_j, D_j)
$$

其中 $D_j$ 为第 $j$ 个评估块的 episode 记录（仅含可部署数据 + 成功与否），$G_j$ 为 Guidance 文档。

## 核心要点

1. **无梯度**：权重始终冻结，改进只发生在 context 层
2. **泛化测量**：每轮使用不重叠的新实例块，确保测量泛化而非记忆
3. **自主诊断**：反思 agent 仅从 episode 记录（无真值标注）推断失败原因
4. **实际效果**：RoboScript 三轮净提升 +18/+8/+4 个 episode（17 个弱任务）

## 代表工作

- [[RoboScript]]: RSI 在操作任务上的实现
- [[Reflexion]]: 相关的 LLM agent 反思框架（单 episode 内）

## 相关概念

- [[World In-Context Learning]]
- [[Task Memory]]
- [[Reflexion]]
