---
type: concept
aliases: [event-driven update, 事件驱动记忆, event-triggered memory]
---

# Event-Driven Memory

## 定义
只在检测到关键事件（物体状态变化、任务阶段切换等）时才更新记忆，而非每个时间步都更新，从而降低计算开销并减少记忆噪声积累。

## 数学形式
$$M_t = \begin{cases} f(M_{t-1}, o_t) & \text{if } \text{event}(o_t) = 1 \\ M_{t-1} & \text{otherwise} \end{cases}$$

## 核心要点
1. 固定频率更新会引入无意义的记忆噪声（无事件帧）
2. 事件检测可基于规则或学习得到
3. 稀疏更新降低 KV-cache 或记忆模块的计算与存储开销

## 代表工作
- [[UniMem]]: 统一记忆与控制的 VLA，使用 event-driven 记忆更新

## 相关概念
- [[MemoryVLA]]
- [[MemER]]
- [[Vision-Language-Action Model]]
