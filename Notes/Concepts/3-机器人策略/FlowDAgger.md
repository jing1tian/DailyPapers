---
type: concept
aliases: ["Flow DAgger"]
---

# FlowDAgger

## 定义
基于 flow matching 的在线模仿学习框架，在 VLA 执行失败时收集人类干预并用 flow 目标微调策略。

## 数学形式
$$\mathcal{L} = \mathbb{E}[||v_\theta(x_t, t) - (x_1 - x_0)||^2]$$

## 核心要点
1. 将 DAgger 的交互式数据收集与 flow matching 策略学习结合
2. 对 VLA 失败的在线恢复更高效

## 代表工作
- 见关联笔记

## 相关概念
- [[DAgger]]
- [[Flow-Matching]]
- [[OopsieRecover]]
