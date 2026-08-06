---
type: concept
aliases: [Recurrent Diffusion Policy]
---

# RDP

## 定义
Recurrent Diffusion Policy：在 Diffusion Policy 中引入循环记忆模块（如 GRU/LSTM）以处理时序依赖的操作策略。

## 核心要点
1. 扩展标准 [[Diffusion Policy]] 加入时序记忆
2. 适合长时序、多阶段操作任务
3. 可用于接触 rich 的动态力控场景

## 代表工作
- [[DPA-IL]]: 与 RDP 对比用于接触 rich 拆卸任务

## 相关概念
- [[Diffusion Policy]]
- [[DP]]
