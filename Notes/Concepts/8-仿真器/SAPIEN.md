---
type: concept
aliases: [SAPIEN Simulator, 具身AI物理仿真器]
---

# SAPIEN

## 定义
专为具身 AI 研究设计的高性能物理仿真平台，支持关节物体交互、灵巧手操作、以及大规模并行化仿真。

## 核心要点
1. 基于 PhysX 物理引擎，支持关节体（articulated objects）精确物理仿真
2. 提供 Python API，支持 PartNet-Mobility 数据集（带关节标注的 3D 物体）
3. 支持 GPU 并行化，可同时运行大量仿真实例
4. 被 ManiSkill 系列 benchmark 用作底层仿真器

## 代表工作
- [[RoboSnap]]: 用 SAPIEN 作为 real-to-sim 重建场景的物理后端

## 相关概念
- [[IsaacLab]]
- [[MuJoCo]]
- [[ManiSkill3]]
