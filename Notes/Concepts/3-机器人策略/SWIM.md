---
type: concept
aliases: [Selective World Model, Behavior-Informed JEPA]
---

# SWIM

## 定义
Hu 等（2026，arXiv 2610.36662，"SWIM: Learning What Matters from Behaviors for JEPA World Models"）提出的行为引导掩码 JEPA 世界模型，通过优先掩码任务相关运动区域（末端执行器轨迹、动作覆盖像素）来提升 latent 表示的动作相关性。

## 核心要点
1. 在 [[V-JEPA]] 框架上改掩码策略：从任务执行轨迹提取行为相关区域，优先掩码这些区域
2. 用 [[CEM]] 在 latent space 上做 MPC 规划
3. 与随机掩码 JEPA 相比，学到的表示对下游动作预测更有效

## 代表工作
- Hu et al. (2026): 原始论文，Boston University

## 相关概念
- [[V-JEPA]]
- [[World Model]]
- [[CEM]]
