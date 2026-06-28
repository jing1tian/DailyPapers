---
type: concept
aliases: [PAI-Bench, PAI Bench]
---

# PAI-Bench

## 定义
面向物理 AI（Physical AI）世界模型的评测基准，覆盖机器人（Robot subset）等多个领域，从视觉质量、物理合理性（physics adherence）、指令跟随等维度对视频生成/世界模型的 rollout 质量打分，用于衡量模型是否同时具备真实感生成能力与符合物理规律的动力学预测能力。

## 核心要点
1. 评测指标通常包含 AVG_PA（物理符合度，Physics Adherence）、AVG_IF（指令跟随，Instruction Following）、AVG_Score（综合总分）等子项，分别考察生成结果是否遵循物理规律、是否准确响应输入指令
2. 区别于只评估视觉真实感的传统视频生成评测，PAI-Bench 更强调生成内容能否作为可靠的世界模型 rollout 用于下游机器人决策
3. 常与 WorldModelBench、DreamGen Bench 等基准搭配使用，共同构成对世界模型生成质量的多角度评测体系

## 代表工作
- [[Kairos]]: 在 PAI-Bench Robot Set 上 AVG_PA 与 AVG_Score 均排名第一，AVG_IF 以远小参数量逼近 14B 级别的 Wan2.2

## 相关概念
- [[WorldModelBench]]
- [[DreamGen]]
