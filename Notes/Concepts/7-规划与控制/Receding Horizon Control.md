---
type: concept
aliases: [渐退视界控制, 滚动时域控制, RHC]
---

# Receding Horizon Control

## 定义

渐退视界控制（RHC）是一种在线优化控制策略：每个时步求解固定时域（Horizon）内的最优动作序列，但只执行序列中的前几步，然后利用最新观测重新规划。该思想是 MPC（模型预测控制）的核心原则。

## 数学形式

在时刻 $t$，求解：

$$
\hat{A}_{t:t+H} = \arg\min_{A} \sum_{k=0}^{H} \mathcal{C}(s_{t+k}, a_{t+k})
$$

执行 $a_t, \ldots, a_{t+\delta-1}$（$\delta < H$），观测 $s_{t+\delta}$ 后重复。

## 核心要点

1. **"执行部分、重新规划"**: 预测 $H$ 步，执行 $\delta$ 步（$\delta < H$），循环滚动。
2. **对扰动鲁棒**: 新观测驱动重规划，累积误差不扩散。
3. **Action Chunking 的控制框架**: 机器人学习中将预测的动作块视为 RHC 的开环实例（$\delta=3, H=7$ 等）。

## 代表工作

- [[WorldDiT]]: 预测 7 步动作块，每次执行前 3 步后重规划
- [[Diffusion Policy]]: 采用 Action Chunking + RHC 推理策略

## 相关概念

- [[Action Chunking]]
- [[Temporal Ensembling]]
- [[MPC]]
