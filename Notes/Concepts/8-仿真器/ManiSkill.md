---
type: concept
aliases: [ManiSkill, ManiSkill2, ManiSkill3, SAPIEN Manipulation Skill]
---

# ManiSkill

## 定义
基于 SAPIEN 物理引擎的开源机器人操作 benchmark 套件，提供多样化灵巧操作任务、标准化评估协议和高质量渲染，支持 GPU 并行仿真。

## 核心要点
1. ManiSkill2 → ManiSkill3 持续演进，任务数量和物理真实性不断提升
2. GPU 并行仿真（SAPIEN + Vulkan），支持大规模 RL 训练
3. 提供 dense/sparse reward，支持 demonstration-based 方法
4. [[COMET]]、[[ContactWorld]] 等 WM/规划工作在 ManiSkill 上做评估

## 代表工作
- [[COMET]]：在 ManiSkill 上测试 object-centric MCTS 规划

## 相关概念
- [[MuJoCo]]
- [[Genesis]]
- [[Isaac Lab]]
- [[SIMPLER]]
