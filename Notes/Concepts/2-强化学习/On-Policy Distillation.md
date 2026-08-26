---
type: concept
aliases: [OPD, on-policy distillation, 在线策略蒸馏]
---

# On-Policy Distillation

## 定义
在模型蒸馏过程中，让 student 模型在自身 rollout 产生的在线数据上接受教师模型监督，从而避免 offline 分布漂移导致的任务能力退化。

## 数学形式
$$\mathcal{L}_\text{OPD} = \mathbb{E}_{s \sim \pi_\text{student}} \left[ D_\text{KL}(p_\text{teacher}(a|s) \| p_\text{student}(a|s)) \right]$$

## 核心要点
1. Offline 蒸馏（SFT）的 student 遇到 off-distribution 状态时性能崩溃
2. On-policy 采样让 student 自己探索状态空间，教师为这些新状态提供标签
3. 与传统 KD 区别：数据分布是 student 的而非 teacher 的

## 代表工作
- [[WAM-OPD]]: 用 OPD 修复 WAM student 在机器人操控中的退化

## 相关概念
- [[GKD]]
- [[Generalized Knowledge Distillation]]
- [[World Action Model]]
