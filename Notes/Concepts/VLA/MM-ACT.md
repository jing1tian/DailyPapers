---
type: concept
aliases: [MM-ACT, 多模态动作生成]
---

# MM-ACT

## 定义

MM-ACT 是一种多模态动作生成 VLA 基线方法，在 LIBERO benchmark 上取得较强的正常光照性能，但与其他 RGB-only 方法一样对视觉退化条件敏感。

## 核心要点

1. 在 LIBERO 正常光照下平均成功率 96.3%，与 ResVLA 并列
2. 在 LL-Severe 极端弱光条件下成功率下降至 69.6%，相比正常光照下降 ~27%
3. 作为 [[EventVLA]] 的主要对比基线之一

## 代表工作

- [[EventVLA]]: 以 MM-ACT 为基线，在 LL-Severe 条件下 EventVLA 超越其 25.9%（95.6% vs 69.6%）

## 相关概念

- [[VLA]]: MM-ACT 所属的方法类别
- [[LIBERO]]: 主要评测 benchmark
