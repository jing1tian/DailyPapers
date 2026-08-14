---
type: concept
aliases: [AgiBot World, AgiBot World Dataset]
---

# AgiBot

## 定义
AgiBot World：大规模机器人操作数据集与评估平台，由 AgiBot 公司发布，包含多种机器人形态（单臂、双臂、移动底盘）的真实遥操作演示数据，设计目标是支持通用 VLA 模型的预训练与评估。

## 数学形式
数据集规格（参考公开资料）：
- 百万量级 episodes，涵盖日常操作任务
- 多模态观测：RGB、深度、本体感知
- 多机器人形态支持

## 核心要点
1. 中国团队主导的大规模机器人数据集，定位对标 OXE（Open-X Embodiment）
2. 支持多传感器模态，适配 MuseVLA 等多感知 VLA 的预训练
3. 提供标准化评估协议，与 [[LIBERO]]、[[SimplerEnv]] 等形成互补
4. 与 [[PaliGemma]] 等 VLM backbone 配合使用

## 代表工作
- [[MuseVLA]]：在 AgiBot World 上训练的多模态感知 VLA

## 相关概念
- [[LIBERO]]: 另一常用机器人操作 benchmark
- [[DROID]]: 大规模真实世界机器人数据集
- [[RoboTwin]]: 双臂操作仿真数据集
