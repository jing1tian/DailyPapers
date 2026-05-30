---
type: concept
aliases: [GROOT, Generalist Robot Manipulation via Object-Centric Abstraction]
---

# GROOT

## 定义
一种基于 object-centric 抽象的通用机器人操作策略，通过学习以物体为中心的表示来提升跨任务跨场景的泛化能力。

## 核心要点
1. 用物体级别（而非像素级别）的抽象表示作为策略输入，减少对视觉细节的依赖
2. 支持 zero-shot 迁移到新场景/新物体
3. 在多个操作任务上展示了强泛化性
4. 被 [[Gaze2Act]] 等论文作为 baseline 对比

## 代表工作
- Shi et al. (2024): GROOT 原始论文
- [[Gaze2Act]]: 以 GROOT 为 baseline

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[OpenVLA]] — 另一类通用操作策略
