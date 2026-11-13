---
type: concept
aliases: [学习世界模型, Neural World Model]
---

# Learned World Model

## 定义

用神经网络学习预测世界状态转移的模型，可在想象中模拟行为结果，潜在地替代物理仿真器用于策略学习或 WICL 排练。

## 核心要点

1. **预测未来观测**：给定当前状态和动作，预测下一帧图像或隐状态
2. **想象中学习**：Dreamer 等方法在世界模型内部训练策略，无需环境交互
3. **WICL 视角**：若学习世界模型语义精度足够，可替代物理仿真器用于 WICL 排练，扩展到无仿真器场景

## 代表工作

- [[WAM]]: World Action Model，学习世界模型并从中导出动作
- [[RoboScript]]: 以物理仿真器为世界，未来可用 Learned World Model 替代

## 相关概念

- [[WAM]]
- [[World In-Context Learning]]
- [[Simulator]]
