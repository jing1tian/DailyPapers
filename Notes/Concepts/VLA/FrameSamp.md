---
type: concept
aliases: [FrameSamp, Frame Sampling]
---

# FrameSamp

## 定义

FrameSamp 是一种简单的 VLA 记忆策略，通过均匀或关键帧采样历史视觉帧作为长期上下文输入，让模型具备有限的历史感知能力，主要用于 RoboMME benchmark 的基线评测。

## 核心要点

1. **简单采样**: 从历史帧序列中等间隔选取若干帧拼入上下文
2. **记忆效率低**: 无法区分重要帧与非重要帧，信息压缩有限
3. **与 TokenDrop 对比**: TokenDrop 在 token 层级丢弃，FrameSamp 在帧层级采样

## 相关概念

- [[TokenDrop]]
- [[RMT]]
- [[MemER]]
- [[RoboMME]]
