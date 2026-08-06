---
type: concept
aliases: [Dynamics Supervision]
---

# DYN

## 定义
DYN：在 VLA 训练中加入动力学预测辅助监督（预测下一帧图像或状态），以增强模型对物理世界变化的理解。

## 核心要点
1. 在动作预测 loss 之外加入 future state prediction loss
2. 提供物理因果关系的内在监督信号
3. 与 UVT 类方法的核心思路相近

## 代表工作
- [[UVT]]: 对比 DYN 等辅助监督方案，提出统一视觉运动目标

## 相关概念
- [[UVT]]
- [[JEPA]]
