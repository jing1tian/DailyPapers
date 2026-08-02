---
type: concept
aliases: [OpenRLHF]
---

# OpenRLHF

## 定义
开源的大规模 RLHF（Reinforcement Learning from Human Feedback）训练框架，支持 PPO、GRPO 等算法，基于 DeepSpeed + Ray 实现大模型分布式 RL 训练。

## 核心要点
1. 支持 PPO、GRPO、DPO 等主流对齐算法
2. 基于 DeepSpeed + Ray 进行多节点分布式训练
3. 支持 LoRA、PEFT 等参数高效方法
4. 常被用作 agentic RL 研究的基线框架

## 代表工作
- 被 [[Molt]] 等 agentic RL 框架列为对比系统

## 相关概念
- [[GRPO]]
- [[PPO]]
- [[PEFT]]
