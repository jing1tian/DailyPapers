---
type: concept
aliases: [MemER, Memory Experience Replay for VLA]
---

# MemER

## 定义

MemER 是一种专为 VLA 设计的记忆增强经验回放机制，将历史经验作为外部记忆存储，在推理或训练时选择性地检索相关历史帧/经验以增强当前决策的历史感知能力。

## 核心要点

1. **记忆库**: 维护历史观测-动作对的记忆缓冲区
2. **检索机制**: 基于相似性从记忆库中检索与当前状态相关的历史经验
3. **VLA 记忆能力**: 专门为长时序依赖任务（计数、遮挡恢复）设计
4. **RoboMME 评测**: 在记忆 benchmark 中与 FrameSamp、TokenDrop、RMT 对比

## 相关概念

- [[FrameSamp]]
- [[TokenDrop]]
- [[RMT]]
- [[经验回放]]
- [[空间记忆]]
