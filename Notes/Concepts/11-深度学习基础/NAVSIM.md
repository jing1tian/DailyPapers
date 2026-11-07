---
type: concept
---

# NAVSIM

## 定义
非反应式驾驶仿真基准，用 EPDMS 指标评估驾驶策略在真实场景重放中的表现。

## 数学形式
$$\text{EPDMS} = \frac{1}{|T|}\sum_t \mathbb{1}[\text{ego safe \& on path}]$$

## 核心要点
1. GuidedWAM 和 VFlash 的主要评测 benchmark
2. 强调无需闭环仿真的开环安全评估

## 代表工作
- 见关联笔记

## 相关概念
- [[GuidedWAM]]
- [[VFlash]]
