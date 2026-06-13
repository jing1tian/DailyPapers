---
type: concept
aliases: [MIKASA-Robo, 部分可观测操作基准]
---

# MIKASA

## 定义
MIKASA-Robo，专为测试机器人策略在**部分可观测**（partially observable）场景下的记忆能力而设计的操作基准，任务需要记住历史视觉信息才能完成。

## 核心要点
1. **部分可观测设计**：目标物体或关键状态在某些时间步不可见
2. **记忆测试任务**：如 RememberColor（记住特定颜色的杯子）、TakeItBack（取走再放回）等
3. **泛化测试**：含 held-out 任务，测试记忆结构的泛化能力

## 代表工作
- [[μVLA]] (2606.12497): 在 MIKASA-Robo 上从 0.42 提升到 0.84

## 相关概念
- [[LIBERO]]
- [[OpenVLA]]
