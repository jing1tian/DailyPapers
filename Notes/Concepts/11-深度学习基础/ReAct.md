---
type: concept
aliases: [Reason + Act, ReAct Prompting]
---

# ReAct

## 定义
将推理（Reasoning）和行动（Acting）交织在 LLM 的 token 序列中，让 agent 在每步先 Thought 再 Action，从环境获取 Observation 后继续推理。

## 数学形式
$$\tau = (t_1, a_1, o_1, t_2, a_2, o_2, \ldots)$$

其中 $t_i$=Thought，$a_i$=Action，$o_i$=Observation，三元组交替生成。

## 核心要点
1. 推理轨迹帮助 agent 追踪计划、处理异常
2. 行动与环境交互获取外部信息（搜索、执行代码等）
3. 比纯 CoT 多了真实反馈回路

## 代表工作
- Yao et al., 2023：ReAct 原始论文
- [[Mosaic]]：多 session agentic RL 中用 ReAct 框架

## 相关概念
- [[CoT]]
- [[ExpeL]]
