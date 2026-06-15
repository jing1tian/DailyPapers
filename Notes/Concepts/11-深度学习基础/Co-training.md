---
type: concept
aliases: [协同训练, Multi-task Co-training, 联合训练]
---

# Co-training（协同训练）

## 定义

在同一训练过程中，将来自不同任务或数据模态的训练流合并，共享模型参数，使模型同时学习多种能力，从而实现正向迁移和知识互补。

## 数学形式

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task1}} + \lambda \mathcal{L}_{\text{task2}}
$$

每次迭代对不同任务 batch 分别做前向计算，梯度累积后统一反向更新。

## 核心要点

1. **正向迁移**：辅助任务提供额外监督信号，改善主任务的特征表示。
2. **灾难性遗忘风险**：若任务梯度方向冲突，可能破坏预训练能力（如 VLM 的视觉-语言对齐）。
3. **提示诱导推理缺口**：VLA co-training 中，动作预测 prompt 可能使模型绕过 co-training 学到的空间先验。
4. **梯度分析**：Projection-Space Similarity（PSS）可量化两任务梯度的协同/对抗关系。

## 代表工作

- [[3DThinkVLA]]: 在 VLA 数据和真实 3D 推理数据上 co-train，发现并解决提示诱导推理缺口问题

## 相关概念

- [[知识蒸馏]]
- [[Multi-Task Learning]]
- [[自蒸馏]]
