---
type: concept
aliases: [Control-Sufficient State, 控制充分状态]
---

# Control-Sufficient State

## 定义
Kairos 提出的世界模型状态表示准则：状态表示不需要保留所有视觉细节，只需保留对具身决策充分必要的变量。

## 数学形式
$$s_t^{cs} = \{v \in S_t \mid v \text{ 影响策略 } \pi \text{ 的输出}\}$$

即从完整状态 $S_t$ 中提取策略相关子集，滤除无关视觉冗余。

## 核心要点
1. 与全保真视觉重建相反，Control-Sufficient State 以"够用"为原则，降低生成难度
2. 在长 horizon 任务中避免无关背景变化导致的误差累积
3. 与 [[JEPA]] 的预测潜变量思想相通

## 代表工作
- [[Kairos]]: Physical AI 世界模型框架，核心设计原则

## 相关概念
- [[JEPA]]
- [[World Model]]
- [[Latent Space]]
