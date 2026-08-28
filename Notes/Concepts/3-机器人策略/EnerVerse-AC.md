---
type: concept
aliases: [EVAC, EnerVerse Action-Conditioned]
---

# EnerVerse-AC (EVAC)

## 定义
基于 UNet 架构的动作条件化世界模型，在 AgiBot World 数据上预训练，用于机器人策略的视频预测和主动学习。

## 核心要点
1. 动作条件化：给定当前帧和动作序列，预测未来视觉观测
2. UNet 架构，适合多分辨率特征融合
3. 在 AgiBot World 大规模数据集上预训练，泛化能力较强

## 代表工作
- [[ConfAL-WM]]: 以 EVAC 为 backbone，结合置信度引导主动学习

## 相关概念
- [[AgiBot World]]
- [[ConfAL-WM]]
- [[World Model]]
