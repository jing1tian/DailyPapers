---
type: concept
aliases: [Chain-of-Thought, CoT推理]
---

# CoT（Chain-of-Thought）

## 定义
让模型在输出最终答案前生成中间推理步骤的提示策略，在 VLA 中用于分解复杂任务为子步骤。

## 核心要点
1. 提示模型逐步推理：观察 → 分析 → 动作计划 → 执行
2. 在 WALL-WM、OneVLA 等 VLA 中加入显式推理链
3. 缺点：增加推理 token 数，实时控制中有 latency 代价

## 代表工作
- [[WALL-WM]]: 用 CoT 做事件边界推理
- [[OneVLA]]: 用 CoT 决定导航/操作分支

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[LLM]]
