---
type: concept
aliases: [LWM, 语言世界模型, Language-based World Model]
---

# Language World Model

## 定义

语言世界模型（LWM）是一类以语言模型为核心、通过预测智能体交互环境下一观测状态来建模环境动态的世界模型，用于推理、规划和智能体 RL 仿真。

## 数学形式

$$
\hat{o}_{t+1} = f_\theta(c, o_{\leq t}, a_{\leq t})
$$

其中 $c$ 为任务上下文，$o_{\leq t}$ 为历史观测，$a_{\leq t}$ 为历史动作，$f_\theta$ 为参数化的语言世界模型。

## 核心要点

1. 以语言/文本为统一接口，支持文本域（MCP、Search、Terminal、SWE）和 GUI 域（Android、Web、OS）的统一仿真
2. 通过 Chain-of-Thought 推理显式建模状态转移逻辑，而非隐式拟合
3. 可作为解耦环境仿真器（替代真实环境）或统一基础模型（智能体预热）
4. 与视频世界模型（以像素为预测目标）互补，适合文本/结构化观测场景

## 代表工作

- [[QwenAgentWorld]]: 首个跨 7 域统一语言世界模型，包含 35B 和 397B MoE 变体

## 相关概念

- [[GSPO]]: 语言世界模型 RL 训练常用算法
- [[Chain-of-Thought Reasoning]]: LWM 推理机制
- [[Mixture-of-Experts]]: LWM 常用的大规模模型架构
