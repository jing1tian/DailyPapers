---
type: concept
aliases: [Diffusion Policy 3D, 3D Diffusion Policy]
---

# DP3

## 定义

DP3（3D Diffusion Policy）是将扩散策略（Diffusion Policy）扩展到 3D 点云输入的机器人操作策略，利用 3D 感知提升空间理解和操作精度。

## 核心要点

1. **3D 感知**: 使用点云作为输入，而非仅 RGB 图像，提升空间感知
2. **扩散模型**: 基于 [[Diffusion Policy]] 框架，保留扩散去噪的动作生成范式
3. **效率**: 轻量 3D 编码器设计，可在实机上实时运行

## 代表工作

- DP3 (Ze et al., 2024): 原始论文，展示了 3D 点云输入对操作任务的提升
- [[AgenticRobots-Survey]]: 将 DP3 列为 3D-aware policy 的代表工作

## 相关概念

- [[Diffusion Policy]]: 基础方法
- [[VLA]]: 同类操作策略的更大规模版本
- [[ACT]]: 同期代表性操作策略
