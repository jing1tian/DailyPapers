---
type: concept
aliases: [课程学习, 渐进式训练, Progressive Training]
---

# Curriculum Learning

## 定义

课程学习（Curriculum Learning）是一种训练策略，模仿人类学习由易到难的过程，按照从简单到复杂的顺序安排训练样本或任务，以提升模型收敛速度和最终性能。

## 数学形式

设训练阶段序列为 $\{S_1, S_2, \ldots, S_K\}$，每阶段使用数据集 $\mathcal{D}_k$ 且满足复杂度递增：

$$
\text{complexity}(\mathcal{D}_1) \leq \text{complexity}(\mathcal{D}_2) \leq \cdots \leq \text{complexity}(\mathcal{D}_K)
$$

多阶段损失通常设计为：

$$
\mathcal{L}_{Stage_k} = \sum_i \lambda_i^{(k)} \mathcal{L}_i
$$

其中权重 $\lambda_i^{(k)}$ 随阶段 $k$ 逐步调整（如退火策略）。

## 核心要点

1. **由易到难**: 先在简单/通用数据上训练，再引入困难/专业数据，避免模型早期过拟合
2. **多阶段设计**: 典型为 2-3 阶段，每阶段有不同的数据分布和损失权重
3. **权重退火**: 辅助损失权重在训练后期逐步衰减，让模型专注于主任务
4. **迁移学习友好**: 结合预训练模型时效果尤为显著，避免灾难性遗忘

## 代表工作

- [[AffordanceVLA]]: 三阶段渐进式课程训练（通用可供性预训练 → 机器人数据联合训练 → 目标任务后训练），$\lambda_{afd}$ 从 0.5 退火至 0.15
- Bengio et al. (2009) "Curriculum Learning": 课程学习的奠基性工作

## 相关概念

- [[Flow Matching]]
- [[Mixture-of-Transformers]]
- [[Action Chunking]]
