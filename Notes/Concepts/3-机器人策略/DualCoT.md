---
type: concept
aliases: [Dual Chain-of-Thought, 双链思考]
---

# DualCoT

## 定义
DualCoT 是一种用于 VLA 长程任务的双层 Chain-of-Thought 框架，分别在高层（任务级别）和低层（动作级别）维护独立的推理链，使 VLA 能够在多个抽象层次上进行有效推理。

## 核心要点
1. 高层 CoT：任务分解、子目标规划（语言描述级）
2. 低层 CoT：具体动作序列推理（传感器-动作级）
3. 双层解耦避免高层长程规划干扰低层精细控制
4. 与 [[MemoryVLA]] 类似，解决 VLA 的长程状态管理问题

## 代表工作
- [[MemoryVLA]]: 借鉴 DualCoT 框架做显式语言记忆增强

## 相关概念
- [[CoT]]
- [[VLA]]
- [[MemoryVLA]]
