---
type: concept
aliases: [VLAC, Vision-Language-Action-Critic]
---

# VLAC

## 定义

VLAC (Vision-Language-Action-Critic) 是一种视觉-语言-动作 Critic 模型，用于估计机器人任务中各子目标的完成进度，为在线 RL 适应提供密集奖励信号。

## 核心要点

1. 输入：子目标起始观测 $o_{start}^k$、结束观测 $o_{end}^k$ 以及子目标描述 $g_k$
2. 输出：子目标进度增量 $\Delta_k(\tau) = C_\phi(o_{start}^k, o_{end}^k, g_k)$
3. 与任务分解结合使用，为每个子目标提供独立进度估计

## 数学形式

$$
\Delta_k(\tau) = C_\phi(o_{start}^k, o_{end}^k, g_k)
$$

## 代表工作

- [[Agentic-VLA]]: 将 VLAC 与 ARS 结合，实现能力感知的课程式奖励

## 相关概念

- [[自适应奖励合成]]
- [[视觉语言动作模型]]
- [[奖励函数]]
