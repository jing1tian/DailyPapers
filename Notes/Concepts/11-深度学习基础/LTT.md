---
type: concept
aliases: ["Learn Then Test", "Learn-Then-Test"]
---

# LTT

## 定义
统计检验框架（Learn Then Test），先用 holdout 数据学习 calibration 参数，再通过多重假设检验给出带 PAC 保证的预测置信区间。

## 数学形式
$$P(\text{error} > \epsilon) \leq \delta, \quad \text{calibrated via LTT}$$

## 核心要点
1. VLA 失败预测论文用它给 failure alarm 加置信保证
2. 把 Bonferroni 校正引入 ML 预测校准

## 代表工作
- 见关联笔记

## 相关概念
- [[STAC]]
- [[FailureSpot]]
