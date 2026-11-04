---
type: concept
aliases: [SAFE, Safe Failure Prediction]
---

# SAFE

## 定义
一种从策略特征学习失败预测器的方法，通过监督信号训练外部分类器判断 VLA 策略在当前状态下是否会失败。

## 核心要点
1. 外部失败预测器，不修改 policy 本身
2. 从 policy 内部特征（如隐状态）学习失败信号
3. 是 CalVLA 中对比的失败检测基线

## 代表工作
- [[CalVLA]]: 与 SAFE 对比，提出 native 置信度 shaping

## 相关概念
- [[CalVLA]]
- [[SAFECAST]]
- [[ActProbe]]
