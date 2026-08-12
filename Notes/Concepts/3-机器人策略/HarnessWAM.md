---
type: concept
aliases: [Harness World Action Model]
---

# HarnessWAM

## 定义
一个在 World Action Model 之上的 agentic 框架层，解决 WAM 有限视野预测与复杂任务全局审议之间的"预测-审议 gap"，支持跨阶段状态维护、执行验证和失败恢复。

## 核心要点
1. **Evidence-Grounded Task State**：维护基于证据的任务状态，跨阶段追踪进度
2. **Capability-Conditioned Executable-Space Projection**：根据 WAM 能力约束可执行动作空间
3. **Progress-Conditioned Event Control**：基于进度触发事件，驱动 agentic 控制流
4. 在 RoboMemArena 和 RoboCerebra 两个新 benchmark 上评测（仅 sim，无真实实验）

## 局限
- 仅有仿真实验，benchmark 为论文自建
- Agentic 推理开销未量化
- 缺少与主流 benchmark（LIBERO、RoboTwin）的直接对比

## 代表工作
- [[HarnessWAM]]: Bridging Prediction and Deliberation in World Action Models (2026)

## 相关概念
- [[WAM]]
- [[LingBot]]
- [[MemoryVLA]]
