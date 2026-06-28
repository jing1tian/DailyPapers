---
type: concept
aliases: []
---

# WorldModelBench

## 定义
用于评测视频/世界模型是否具备物理一致动力学预测能力的基准，覆盖多个领域（机器人操作、自动驾驶、人类活动等），通过判断模型生成的 rollout 是否遵循基本物理规律（如刚体碰撞、重力、因果关系）来给生成式世界模型打分，区别于只看视觉真实感的传统视频生成评测。

## 核心要点
1. 评测重点是"动力学合理性"而非"画面好看"——专门设计了违反物理规律的负样本来检验模型是否学到了真实动力学，而不是纯粹的视觉先验
2. 覆盖多领域场景，用于横向比较不同世界模型在 embodied / physical AI 场景下的通用性
3. 常被新提出的统一世界模型（如把视觉生成、动作预测打包在一起的模型）用作"是否真的理解物理"的检验标准

## 代表工作
- [[Kairos]]: 在 Embodied World Model Benchmarks / General World Model Benchmarks 中使用 WorldModelBench 类基准验证长视野 rollout 的物理一致性

## 相关概念
- [[LIBERO]]
- [[DreamGen]]
