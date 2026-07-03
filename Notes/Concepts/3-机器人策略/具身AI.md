---
type: concept
aliases: [具身人工智能, Embodied AI, 具身智能]
---

# 具身 AI

## 定义

具身 AI（Embodied AI）是指将 AI 系统嵌入到具有物理实体（机器人身体）中，使其能够通过感知-动作循环与物理环境交互的研究范式，区别于纯语言或图像处理的"离身"AI。

## 核心要点

1. **感知-动作闭环**: 系统通过传感器感知环境，通过执行器作用于环境，形成持续的交互循环
2. **任务定义**: 设计策略 $\pi$，将初始构型（initial configuration）转化为目标构型（target configuration）
3. **三要素**: 机器人本体（agent）、任务相关物体（objects of interest）、周围环境（ambient environment）
4. **核心挑战**: 泛化能力、物理理解、多模态感知融合

## 代表工作

- [[WAMTutorial]]: 从世界模型到世界动作模型的教程，系统梳理具身 AI 的预测与控制框架

## 相关概念

- [[世界模型]]
- [[世界动作模型]]
- [[闭环策略]]
