---
type: concept
aliases: [EmPACT, Embedded Physics-Aware Control Transformer]
---

# EmPACT

## 定义
将物理约束（如力、刚体动力学）嵌入 VLA/操作策略的方法，通过显式物理先验改善 VLA 在接触丰富任务中的动作一致性。

## 核心要点
1. 与 [[ForceVLA]]、[[PhysVLA]] 属于同一"物理约束 VLA"方向
2. 在 VLA 生成的动作上叠加物理可行性约束，减少运动学/动力学违规
3. [[PhysVLA]] 论文 method_names 中出现 EmPACT，作为该方向的前置工作

## 代表工作
- [[PhysVLA]]：与 EmPACT 在物理约束 VLA 方向对比

## 相关概念
- [[PhysVLA]]
- [[OpenVLA]]
- [[CogACT]]
