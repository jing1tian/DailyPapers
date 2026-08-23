---
type: concept
aliases: [Agentic Evaluation, Agent评估, 智能体评估]
---

# Agentic 评估

## 定义
用分层 Agent 替代固定 rubric，根据评估上下文动态分解问题、调用专用工具获取证据，并生成可追溯推理链的评估范式。

## 核心要点
1. **上下文感知**: 评估 Agent 理解每个案例的具体语义，而非应用统一指标
2. **层次分解**: 将高层评估问题分解为可测量的子问题，分配给专用子 Agent
3. **证据聚合**: 父 Agent 验证子 Agent 的结果，生成 [[证据树|Evidence Tree]]
4. **透明可追溯**: 最终评分可追溯到每个子问题的具体失败原因

## 数学形式
无统一数学形式；核心思想是将评估函数 $f(\text{case})$ 分解为：
$$f(\text{case}) = \text{Aggregate}\left(\{g_i(\text{sub-question}_i)\}_{i=1}^{N}\right)$$
其中 $g_i$ 是第 $i$ 个子 Agent 的评估函数。

## 代表工作
- [[HarnessEval-W]]: 首个将 Agentic Evaluation 用于交互式世界模型评估的框架

## 相关概念
- [[技能路由]]
- [[证据树]]
- [[交互式世界模型]]
