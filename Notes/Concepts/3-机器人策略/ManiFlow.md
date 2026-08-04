---
type: concept
aliases: [ManiFlow]
---

# ManiFlow

## 定义
基于 Conditional Flow Matching 的机器人操作策略，利用耦合传输（coupled transport）来提高 action 生成效率，减少推理步数（NFE）。

## 核心要点
1. 用历史 action 初始化流而非从高斯噪声出发，缩短 ODE 积分路径
2. 属于 Temporal Policy 系列方法的先驱之一

## 代表工作
- [[Temporal-Policy]]: 引用 ManiFlow 作为 coupled transport 的先驱

## 相关概念
- [[CFM]]
- [[Flow-Matching]]
- [[Diffusion Policy]]
