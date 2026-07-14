---
type: concept
aliases: [PARTNR Benchmark]
---

# PARTNR

## 定义
评测 LLM-based 具身 agent 在协作 household 任务中规划能力的 benchmark，任务在 Habitat 仿真器中运行，要求两个 agent 合作完成整理、移动物品等任务。

## 核心要点
1. 基于 Habitat 仿真器和 HSSD 3D 场景数据集
2. 测试多智能体协调、任务分工、通信效率
3. 包含部分可观测性，agent 需要通过通信对齐各自的世界模型
4. 用于评测 LLM planner 在多智能体设置下的能力

## 代表工作
- [[Embodied Multi-Agent Coordination by Aligning World Models Through Dialogue]]：在 PARTNR 上验证对话对齐 world model 的有效性

## 相关概念
- [[Habitat]]
- [[ToM]]
