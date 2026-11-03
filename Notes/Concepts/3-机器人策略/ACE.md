---
type: concept
aliases: [ACE Robot, Any-point Trajectory Model]
---

# ACE

## 定义
Unitree 提出的机器人操作框架，通过跨模态对比学习对齐人类视频与机器人动作，用于从大规模人类操作数据中迁移操作技能。

## 核心要点
1. 用对比学习对齐人类手部轨迹与机器人末端执行器轨迹
2. 支持 zero-shot 迁移：无需机器人演示即可从人类视频学习
3. 常作为 human-to-robot transfer 方法的 baseline

## 代表工作
- [[Nexus-VLA]]: 在层次化语义 latent 空间设计中参考了 ACE 的跨模态对齐思路

## 相关概念
- [[VLA]]
- [[DP]]
