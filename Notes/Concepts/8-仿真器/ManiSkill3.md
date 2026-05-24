---
type: concept
aliases: [ManiSkill 3, SAPIEN ManiSkill]
---

# ManiSkill3

## 定义
基于 SAPIEN 物理引擎的机器人操作仿真基准（第三版），支持 GPU 并行仿真、多种操作任务和视觉观测，是评估 manipulation 策略的主流仿真平台之一。

## 核心要点
1. GPU 并行仿真，训练效率远超 CPU-only 引擎
2. 涵盖抓取、堆叠、装配等多种操作任务
3. 支持点云、RGB-D 等多模态观测
4. 被 RoboFlow4D、Diffusion Policy 等工作用作基准

## 代表工作
- [[RoboFlow4D]]: 在 ManiSkill3 上评估 3D flow WM

## 相关概念
- [[MuJoCo]]
- [[RoboTwin]]
- [[IsaacLab]]
