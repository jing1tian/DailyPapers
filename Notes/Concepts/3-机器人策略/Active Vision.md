---
type: concept
aliases: [主动视觉, 主动视角选择, Active Camera Control]
---

# Active Vision（主动视觉）

## 定义

机器人或 AI 智能体主动控制相机位置或视角，以获取更好的观测信息（解决遮挡、提高感知质量），而非被动依赖固定相机。

## 核心要点

1. 智能体需要感知到当前视角不足（如遮挡），并决策移动相机
2. 与[[具身智能]]的主动感知能力密切相关
3. 需要模型具备空间推理和自我定位能力

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: GPT-6-Astra 在花插任务中主动重新定位相机以解决遮挡，单臂失败但双臂 Piper 通过主动视觉成功完成任务

## 相关概念

- [[Embodied AI|具身智能]]
- [[Zero-shot Generalization]]
