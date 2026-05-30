---
type: concept
aliases: [FLARE, Future-conditioned Language-Action Representation]
---

# FLARE

## 定义
一种用未来帧预测（世界模型）辅助 VLA 策略学习的方法，通过预测未来视觉状态来提升 VLA 的操作规划能力。

## 核心要点
1. 将世界模型（未来帧预测）作为辅助任务注入 VLA 训练
2. 属于两阶段方法：先预测未来帧，再用预测帧辅助动作生成
3. 被 [[DUST]] 作为 baseline 对比，DUST 改进了其模态分离设计

## 代表工作
- FLARE 原始论文
- [[DUST]]: 超越 FLARE 的双流扩散世界模型 VLA

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[扩散世界模型]]
- [[DUST]] — 后续改进工作
