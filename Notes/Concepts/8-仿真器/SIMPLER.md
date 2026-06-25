---
type: concept
aliases: [SIMPLER, SimplerEnv, SIMPLER Environment]
---

# SIMPLER

## 定义

SIMPLER（SimplerEnv）是一个专门用于评测 VLA 模型的视觉机器人操作评测环境，在仿真中重建真实机器人操作场景，通过视觉逼真度提升仿真到真实的迁移可靠性，降低评测对真实机器人的依赖。

## 核心要点

1. **视觉真实感**: 用高质量渲染仿真真实机器人操作场景（Google Robot、WidowX）
2. **真实数据对齐**: 仿真评测结果与真实机器人评测结果高度相关
3. **标准评测基准**: 成为 VLA 论文（OpenVLA、SAGPruning 等）的标准评测基准之一
4. **任务覆盖**: 包含 Google Robot 上的物体操作任务和 BridgeData 场景

## 代表工作

- Li et al. 2024: SIMPLER/SimplerEnv 原始论文
- [[SC3-Eval]]: 沿用 SIMPLER 提出的 [[MMRV (Mean Maximum Rank Violation)|MMRV]] 指标衡量视频世界模型评测器对策略排序一致性的还原能力，并与 SIMPLER 等物理仿真路线在真实世界基准上做对比

## 相关概念

- [[OpenVLA]]
- [[SimplerEnv]]
- [[ManiSkill3]]
- [[Robosuite]]
- [[MMRV (Mean Maximum Rank Violation)]]
- [[Pearson Correlation]]
