---
type: concept
---

# OLAF

## 定义
VLA 在线自适应学习框架，在推理时检测分布偏移并自适应调整策略。

## 数学形式
$$\pi' = \arg\min_\pi \mathcal{L}_{adapt}(\pi, \mathcal{D}_{online})$$

## 核心要点
1. 无需离线重训练即可在线恢复
2. 与 OopsieRecover 在 VLA 纠错上是对比方法

## 代表工作
- 见关联笔记

## 相关概念
- [[DAgger]]
- [[OopsieRecover]]
