---
type: concept
aliases: [TokenDrop, Token Dropping for Memory]
---

# TokenDrop

## 定义

TokenDrop 是一种 VLA 记忆压缩策略，通过随机或策略性丢弃历史 token 来控制上下文长度，在保留部分历史信息的同时限制计算开销。

## 核心要点

1. **Token 级别操作**: 在 token 序列层面压缩历史，而非帧层面
2. **随机或重要性驱动**: 可以随机丢弃或基于重要性分数选择丢弃哪些 token
3. **RoboMME 评测**: 作为记忆 benchmark 的基线策略之一

## 相关概念

- [[FrameSamp]]
- [[RMT]]
- [[MemER]]
- [[RoboMME]]
