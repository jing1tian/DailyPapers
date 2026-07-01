---
type: concept
aliases: [GE-Act, GE-Action]
---

# GE-Act

## 定义

一种结合生成式视觉预测与动作策略的机器人操作方法，通过显式视觉生成辅助动作预测，在 LIBERO 等 benchmark 上表现优异。

## 核心要点

1. **生成-执行范式**: 先生成目标视觉状态，再以此为条件预测执行动作
2. **性能**: 在 LIBERO 域内及 OOD 迁移场景中是强基线（LIBERO-Plus-Spatial 87.5%）
3. **对比位置**: A2World-policy（88.5%）在 OOD 迁移上小幅超越 GE-Act

## 代表工作

- [[A2World]]: 将 GE-Act 作为策略对比基线（OOD 迁移成功率 87.5% vs A2World 88.5%）

## 相关概念

- [[Action-Conditioned World Model]]
- [[LIBERO]]
- [[π0]]
