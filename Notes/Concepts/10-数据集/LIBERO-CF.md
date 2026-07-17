---
type: concept
aliases: [LIBERO Counterfactual, LIBERO CF Benchmark]
---

# LIBERO-CF

## 定义

基于 [[LIBERO]] 环境构建的首个 VLA 反事实语言跟随评测基准，包含 50 个任务，专门测试 VLA 在语言指令与视觉场景不一致时的语言遵循能力。

## 数学形式

两个核心指标：

- **Grounding Rate**: 机器人是否接触到语言指定的目标物体
- **Success Rate**: 任务是否完成

## 核心要点

1. 四套任务：CF-Spatial（15 任务）、CF-Object（10 任务）、CF-Long（10 任务）、CF-OOD（15 任务）
2. 核心设计：场景中同时存在训练任务对象（诱导视觉捷径）和语言指定的反事实目标
3. 区分 Faithful（语言正确）和 Biased（语言指向训练对象）两类条件进行评测

## 代表工作

- [[CAG]]: 提出 LIBERO-CF 基准，UNC Chapel Hill，2026

## 相关概念

- [[LIBERO]]
- [[VLA（视觉-语言-动作模型）]]
- [[Counterfactual Action Guidance]]
