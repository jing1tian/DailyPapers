---
type: concept
aliases: [Dance GRPO, DanceGRPO Framework]
---

# DanceGRPO

## 定义
DanceGRPO 是一种将 GRPO（Group Relative Policy Optimization）策略优化算法应用于扩散/流匹配视频生成模型的在线强化学习微调方法，用于利用黑盒奖励优化视频生成质量。

## 核心要点
1. **反向过程 RL**：在去噪轨迹上进行策略优化，属于反向过程 RL 类方法
2. **组相对优势**：使用同 prompt 内的相对奖励计算优势估计，继承 GRPO 框架
3. **信用分配问题**：由于奖励在完整生成后才可得，单步信用分配较弱，是此类方法的主要局限

## 相关方法对比

与 [[DiffusionART]] 对比（T2V, TA:VQ:MQ=7:2:1, step 400）：
- DanceGRPO: VQ=3.922, MQ=1.206
- DiffusionART: VQ=8.970, MQ=4.224

## 代表工作
- [[DiffusionART]]: 在 T2V 和 I2V 任务上将其作为对比基线

## 相关概念
- [[FlowGRPO]]: 同属反向过程 GRPO 类方法
- [[GRPO]]: 基础算法框架
