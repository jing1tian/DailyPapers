---
type: concept
aliases: [Cross-Embodiment Learning, 跨本体学习]
---

# Cross-Embodiment

## 定义
跨机器人本体（不同臂型、不同自由度、不同传感器配置）联合训练策略或世界模型的范式：通过统一的数据组织方式（如统一动作空间、统一观测编码）把来自不同机器人平台的数据汇聚成一份大规模训练集，目的是让模型学到的表征/动力学不局限于单一本体，提升跨平台部署和数据效率。

## 核心要点
1. 核心难点是不同本体的动作空间、自由度、传感器模态不一致，需要设计统一的数据 schema 或动作表征才能联合训练
2. 数据课程（data curriculum）常按"开放世界视频 → 人类行为数据 → 机器人交互数据"的渐进式顺序组织，让模型先学通用动力学再细化到具体本体
3. 是当前大规模机器人基础模型（VLA / WAM）数据策略的主流思路之一，代表性数据集如 Open X-Embodiment

## 代表工作
- [[Kairos]]: 用 Cross-Embodiment Data Curriculum 组织开放世界视频、人类行为数据和机器人交互数据，作为 Native Pre-training Paradigm 的核心数据策略

## 相关概念
- [[RoboNet]]
- [[BridgeData]]
- [[VLA]]
