---
type: concept
aliases: [Sky RL, SkyRL-Agent]
---

# SkyRL

## 定义
一个面向 LLM agent RL 训练的开源框架，支持 training-rollout 解耦架构，允许多 turn agent 在沙盒环境中生成 rollout 轨迹，再异步送回训练侧更新策略。

## 数学形式

多轮 agent RL 目标（PPO/GRPO 变体）：
$$\mathcal{L} = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \min\left(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t\right)\right]$$

其中 $\tau$ 为多步 agent 交互轨迹，$r_t = \pi_\theta / \pi_{\text{old}}$ 为概率比。

## 核心要点
1. Training-rollout 解耦，支持分布式异步采样
2. 支持 rootless 沙盒（代码执行隔离），适合工具调用 agent
3. 与 VeRL、NeMo 等训练后端兼容
4. ProRL Agent 论文中将其作为对比基线

## 代表工作
- [[ProRL]]: 将 SkyRL 作为 baseline 对比

## 相关概念
- [[VeRL]]
- [[GRPO]]
- [[DAPO]]
