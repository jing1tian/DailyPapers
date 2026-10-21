---
type: concept
aliases: [star-VLA, StarVLA]
---

# starVLA

## 定义
starVLA 是一种无动作预训练的 VLA（Vision-Language-Action）基线方法，在 RoboTwin 和 LIBERO 等基准上作为对比方法。

## 核心要点

1. **无动作预训练**: 不使用大规模机器人动作数据预训练
2. **泛化能力有限**: 在随机化场景下成功率极低（RoboTwin 随机 3.16%），说明纯 VLA 对外观变化敏感
3. **作为 WAM 方法的基线**: 代表"传统 VLA"在泛化任务上的上限

## 代表工作
- [[PrismWAM]]: 将 starVLA 作为无预训练基线对比，随机化场景下 53.16% vs 3.16%

## 相关概念
- [[World Action Model]]
- [[RGB-based WAM]]
