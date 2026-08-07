---
type: concept
aliases: [MemoryBench Benchmark, 记忆操纵基准]
---

# MemoryBench

## 定义

评估单臂机器人策略在记忆依赖操纵任务上能力的仿真 Benchmark，包含 3 个任务（Reopen Drawer、Put Block Back、Rearrange Block），要求策略记忆初始状态并在后续操作中利用该信息。

## 核心要点

1. **任务数量**：3 个任务，覆盖"关抽屉→重开"、"放回物块"、"重排积木"等记忆依赖场景
2. **单臂平台**：与 [[RMBench]]（双臂）互补，共同验证记忆机制在不同机器人形态上的泛化
3. **评估指标**：任务成功率（Success Rate）

## 代表工作

- [[BridgeVLA++]]: 在 MemoryBench 上达到 99.7 ± 0.3% 的成功率

## 相关概念

- [[RMBench]]
- [[时空记忆]]
- [[RLBench]]
