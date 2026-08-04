---
type: concept
aliases: [DiffusionDrive]
---

# DiffusionDrive

## 定义
将扩散模型用于自动驾驶 planning 的端到端方法，通过扩散过程生成多模态轨迹，在 NAVSIM/nuPlan 等 benchmark 上有强竞争力。

## 核心要点
1. 扩散过程生成 ego 轨迹分布，处理多模态不确定性
2. 端到端训练，无需显式预测中间表示
3. Auto-JEPA 在 NAVSIM 上与 DiffusionDrive 对比

## 代表工作
- [[Auto-JEPA]]: 在 NAVSIM 上与 DiffusionDrive 对比

## 相关概念
- [[NAVSIM]]
- [[Diffusion Policy]]
- [[DriveWorld]]
