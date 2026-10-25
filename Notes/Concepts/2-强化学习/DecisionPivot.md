---
type: concept
aliases: [Decision Pivot, 决策枢纽训练]
---

# DecisionPivot

## 定义
DecisionPivot：StartLux-Decision 论文中提出的第三阶段 RL 训练策略，通过推理监督和强化学习为 AutoThink 模型注入显式推理能力，使其能在不确定时触发 chain-of-thought。

## 数学形式
$$\mathcal{L} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{RL}}(\text{calibration reward})$$

结合监督推理轨迹和置信度校准奖励进行联合训练。

## 核心要点
1. 三阶段训练的最后一步：先建立基础决策能力 → 任务适应 → DecisionPivot 注入推理
2. 使用校准奖励（calibration reward）惩罚过度自信和过度怀疑
3. 是 StartLux-Decision AutoThink 变体的关键区别

## 代表工作
- [[StartLux-Decision]]: 提出 DecisionPivot 的论文

## 相关概念
- [[GRPO]]: 类似的 RL 训练方法
- [[RLVR]]: 同类可验证奖励 RL 框架
