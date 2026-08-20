---
type: concept
aliases: [NVIDIA PhysX, PhysX Engine]
---

# PhysX

## 定义
NVIDIA 开发的实时物理仿真引擎，支持刚体动力学、布料、流体、关节约束等，是 Isaac Sim、ManiSkill3 等机器人仿真平台的底层物理后端。

## 核心要点
1. GPU 加速的并行物理仿真，支持大规模并行环境
2. 提供精确的接触力、摩擦模型，适合机器人操作仿真
3. 在 EmbodiedGen V2 等系统中用于验证生成资产的物理合理性
4. 是 NVIDIA Isaac Sim / Isaac Lab 的核心组件

## 代表工作
- [[EmbodiedGen-V2]] (2607.07459): 用 PhysX 验证生成的 sim-ready 3D 资产

## 相关概念
- [[ManiSkill3]]
- [[SAPIEN]]
