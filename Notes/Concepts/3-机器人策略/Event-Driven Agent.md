---
type: concept
aliases: [Event-Driven Agent, 事件驱动智能体, Event-Driven Embodied Agent]
---

# Event-Driven Agent

## 定义

Event-Driven Agent 是一种用于长时序机器人任务的智能体框架，通过 Monitor-Memory-Planner 三层架构持续追踪任务状态，仅在关键条件变化时触发 LLM 重规划，VLA 策略以当前子任务为条件高频执行。

## 数学形式

**Monitor** 生成字幕:
$$C_t = M(W_t, q_t)$$

**Memory Manager** 更新事件与触发标志:
$$(E_t, z_t) = G(B_t, H_{<t}, X_t, g), \quad z_t \in \{0,1\}$$

**Planner** 初始化子任务序列:
$$Q^{(0)} = P_{plan}(g, s_0, S)$$

## 核心要点

1. **Monitor**: 每 2 秒观测窗口生成结构化字幕，轻量实时运行。
2. **Memory Manager**: 将字幕转为持久事件（追加到事件列表），仅当任务进度或可行性发生变化时置 $z_t=1$。
3. **Planner**: 被 $z_t=1$ 触发时重新评估子任务序列，支持四种动作：维持/推进/等待/重生成。
4. **等待机制**: 当子任务临时不可行时（如目标物体被移除），系统等待条件恢复后自动续接，避免无效规划。
5. 通过持久事件存储实现跨子任务的状态记忆（如钥匙位置在其他任务期间被记录）。

## 代表工作

- [[MR0]]: 用于桌面操作的事件驱动智能体，演示跨任务记忆和条件等待两个定性实验

## 相关概念

- [[Vision-Language-Action]]
- [[Action Chunking]]
- [[Agentic Robot]]
