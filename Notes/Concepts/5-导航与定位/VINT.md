---
type: concept
aliases: [Visual Navigation Transformer, ViNT]
---

# VINT

## 定义
Visual Navigation Transformer，一种通用视觉导航基础模型，以图像目标为条件，在不同机器人平台上实现泛化的目标导向导航。

## 核心要点
1. 统一的视觉导航 foundation model，支持跨机器人迁移
2. 输入：当前观测图像 + 目标图像，输出：导航动作
3. 在多个真实机器人平台和数据集上训练（GNM 数据集）
4. 是 [[GNM]]（General Navigation Model）框架的核心实现

## 代表工作
- [[UniWM]] (2510.08713): 以 VINT 作为对比基线评估统一导航世界模型

## 相关概念
- [[GNM]]
- [[NoMaD]]
- [[World Model]]
