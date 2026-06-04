---
type: concept
aliases: [推理 Dropout, Reasoning Dropout Strategy]
created: 2026-06-04
---

# Reasoning Dropout

## 定义

一种训练策略：在训练时以概率 $p_{cot}$ 随机将样本切换为"含 CoT"或"无 CoT"条件，使模型内化推理模式而非依赖显式推理链，推理时可直接输出动作。

## 数学形式

$$
\text{condition} = \begin{cases} \text{CoT mode} & \text{with prob } p_{cot} \\ \text{no-CoT mode} & \text{with prob } 1 - p_{cot} \end{cases}
$$

## 核心要点

1. **内化而非依赖**: 模型被迫从有无 CoT 两种条件中学习，将推理模式压缩进骨干表示
2. **噪声缓解（CoT 污染）**: 噪声自动标注的 CoT 字段（如 Gripper 坐标）在 Dropout 后负面影响从 -5.6 缩减至 -0.8
3. **测试时灵活性**: 推理时固定使用无 CoT 模式，避免自回归推理链的误差累积
4. **与 Teacher Forcing 对比**: Teacher Forcing 使用真实 token 作为上下文；Reasoning Dropout 则随机丢弃整个推理前缀

## 代表工作

- [[ERVLA]]: 在具身 CoT 训练中引入 Reasoning Dropout，缓解 CoT 污染并实现无推理链的鲁棒动作预测

## 相关概念

- [[Chain-of-Thought Reasoning]]
- [[AR-Forcing]]
- [[Embodied Forcing]]
