---
type: concept
aliases: [Object World Model, Composable World Model]
---

# OWM (Object World Model)

## 定义
基于对象槽（object slot）的可组合世界模型，将场景分解为独立的对象表示，并学习对象间的转换规则。

## 核心要点
1. 用 slot attention 提取 per-object 表示
2. 对象状态转换独立建模（composable transitions）
3. 支持 loop detection 用于抽象规则推理（ARC）

## 代表工作
- [[ARC Slots]]: ETH/Tübingen 将 OWM 用于 ARC 抽象推理

## 相关概念
- [[WAM-Survey]]
- [[WMSD]]
