---
type: concept
aliases: [LIBERO PRO, LIBERO OOD Benchmark]
---

# LIBERO-PRO

## 定义

基于 [[LIBERO]] 的 OOD 鲁棒性评估 benchmark，通过引入 4 类系统性扰动（新物体/新位置/新语义/新任务），测试 VLA 模型在训练分布外场景的泛化能力。

## 数学形式

4 类 OOD 扰动类型：

| 扰动类型 | 英文 | 内容 |
|----------|------|------|
| 新物体 | Obj | 替换为训练中未见的物体 |
| 新位置 | Pos | 改变物体的初始位置 |
| 新语义 | Sem | 改变语言指令的语义 |
| 新任务 | Task | 引入训练中未见的新任务 |

评估覆盖 LIBERO 的 4 个套件（Spatial/Object/Goal/Long），共 4×4=16 种组合。

## 核心要点

1. 专门设计用于评估 VLA 模型的 OOD 泛化能力（标准 LIBERO 仅测 in-distribution 性能）
2. 发现 VLA-0 在位置扰动（Pos）上几乎完全失败（接近 0%），说明对精确空间位置过拟合严重
3. CLAP 2B 在 LIBERO-PRO 平均 +11.1 pts vs VLA-0，语言计划显著提升 OOD 鲁棒性
4. 与 [[LIBERO]] 结合使用，提供 in-distribution 和 OOD 的完整评估

## 代表工作

- [[CLAP]]: 在 LIBERO-PRO 上系统评估了 Language-Action Grounding 对 OOD 鲁棒性的提升

## 相关概念

- [[LIBERO]]
- [[VLA（视觉-语言-动作模型）]]
