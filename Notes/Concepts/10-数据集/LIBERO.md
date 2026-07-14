---
type: concept
aliases: [LIBERO Benchmark, LIBERO-90]
---

# LIBERO

## 定义

LIBERO 是一个语言条件机器人操作 benchmark，包含 130 个任务组成的套件，设计用于评估策略在长时序、语言指令引导下的操作能力。

## 核心要点

1. **语言条件**: 每个任务对应自然语言指令，评估语言理解与操作执行的联合能力。
2. **多样套件**: 包含 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 和 LIBERO-Long 等子集，以及 LIBERO-90（90 个任务用于预训练）。
3. **长时序挑战**: 部分任务需要多步骤顺序操作，测试策略的时序规划能力。
4. **仿真环境**: 基于 robosuite / MuJoCo，包含多个桌面操作场景。

## 代表工作

- [[GR00T-N1.7]]: 在 LIBERO-90 上进行预训练与评估
- [[FlowDAgger]]: 用 LIBERO-90 task 57 验证对 GR00T N1.7 的适应能力

## 相关概念

- [[LIBERO-10]]
- [[VLA（视觉-语言-动作模型）]]
- [[模仿学习]]
