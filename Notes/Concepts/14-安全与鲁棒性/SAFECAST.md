---
type: concept
aliases: [SAFECAST]
---

# SAFECAST

## 定义
基于 rollout 标签学习的 VLA 失败预测方法，通过在成功和失败 rollout 上训练分类器来预测执行失败。

## 核心要点
1. 需要成功/失败 rollout 数据来训练分类器
2. 属于外部失败预测器，与 policy 解耦
3. 是 CalVLA 中对比的失败检测基线之一

## 代表工作
- [[CalVLA]]: 与 SAFECAST 对比，提出不需要外部分类器的 native 置信度方案

## 相关概念
- [[CalVLA]]
- [[SAFE]]
- [[ActProbe]]
- [[INSIGHT]]
