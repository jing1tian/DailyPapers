---
type: concept
aliases: [Residual Policy]
---

# ResiP

## 定义
残差策略（Residual Policy）的一种实现，在基础 chunk policy 输出上叠加残差修正，用于动态调整执行步长。

## 核心要点
1. 基础策略提供主动作序列，残差头提供修正信号
2. 用于 DEHP 中的动态 execution horizon 预测
3. 轻量级，不需要重新训练整个 policy

## 代表工作
- [[DEHP]]: 用 ResiP 实现 chunk-based policy 的动态执行

## 相关概念
- [[Autoregressive Policy]]
- [[Action Diffusion]]
