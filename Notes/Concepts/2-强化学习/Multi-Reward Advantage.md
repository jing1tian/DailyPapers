---
type: concept
aliases: [多奖励优势归一化, Multi-Reward Advantage Normalization]
---

# Multi-Reward Advantage（多奖励优势归一化）

## 定义

强化学习后训练中，将多个异质奖励信号（量纲、方差各不相同）各自独立归一化后加权融合为单一优势估计，用于策略梯度更新。

## 数学形式

$$
\hat{A}^{(i)} = \sum_r w_r \frac{R_r(x_0^{(i)}, c) - \mu_r}{\sigma_r + \delta}
$$

**符号说明**：
- $R_r(x_0^{(i)}, c)$：第 $r$ 个奖励函数对第 $i$ 个生成样本的评分
- $\mu_r, \sigma_r$：组内均值和标准差（对当前 batch/group 统计）
- $w_r$：第 $r$ 个奖励的权重系数
- $\delta$：数值稳定小常数

## 核心要点

1. **量纲解耦**: 各奖励独立标准化，避免数值范围大的奖励主导梯度
2. **组内归一化**: 在同一提示的生成组内做统计，去除提示相关的基准差异（类似 [[GRPO]] 的 advantage 归一化）
3. **加权融合**: 允许为不同维度（美学、物理合理性、任务完成）设置不同权重
4. **应用场景**: [[LingBot-Video]] 使用六维奖励（视觉质量、文视对齐、动态程度、运动连贯性、人体动作一致性、物理合理性）

## 代表工作

- [[LingBot-Video]]: 六维专用奖励模型 + Multi-Reward Advantage 归一化，通过 [[GRPO]] 对扩散模型做 RLHF

## 相关概念

- [[GRPO]]
- [[FlowGRPO]]
- [[Flash-GRPO]]
