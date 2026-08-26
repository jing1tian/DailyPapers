---
type: concept
aliases: [behavior intent, intent distillation, 行为意图蒸馏]
---

# Behavior Intent Distillation

## 定义
用 VLM 从专家演示中自动推断每个动作段服务的局部目标（意图），将意图标注作为隐式监督信号加入 VLA 训练，使策略学到"为什么做"而不只是"做什么"。

## 核心要点
1. BC 只学"执行了哪个动作"，不学"这个动作的局部目的"
2. 意图作为 latent 中间表示，推理时不需要意图输入
3. 意图标注质量依赖 VLM 对操控场景的理解能力

## 代表工作
- [[ActWithIntent]]: 为 VLA 引入行为意图蒸馏

## 相关概念
- [[Behavior Cloning]]
- [[Hierarchical Policy]]
- [[VLA]]
