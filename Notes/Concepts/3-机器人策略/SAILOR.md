---
type: concept
aliases: [SAILOR planner]
---

# SAILOR

## 定义
一种基于世界模型的机器人规划器（model-based planner），通过在 world model 中进行 rollout 来评估和选择动作序列，与 VLM-guided planning 方法（如 [[WAP]]）对比使用。

## 核心要点
1. 通过 WM rollout 做 lookahead planning，不需要人工定义 reward 函数
2. 作为 [[WAP]]（World Action Planner）等方法的对比基线
3. 代表 model-based planning 在 robot manipulation 中的早期工作

## 相关概念
- [[WAP]]
- [[MPPI]]
