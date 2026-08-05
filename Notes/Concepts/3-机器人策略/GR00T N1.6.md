---
type: concept
aliases: [GR00T N1.6, GROOT N1.6, Isaac GR00T N1.6]
---

# GR00T N1.6

## 定义

NVIDIA 发布的具身基础模型，基于大规模多模态数据预训练，支持多类机器人操作任务的零样本和少样本迁移。

## 核心要点

1. 参数量约 3B，包含视觉语言理解和动作生成两个核心部分
2. 利用大规模具身数据（合成数据 + 真实数据）进行预训练
3. 在 LIBERO 等操作基准上表现优异（Spatial/Object/Goal/Long 均值约 97.0%）
4. 支持多种机器人形态和任务类型

## 代表工作

- [[SG-WAM]]: 以 GR00T N1.6 作为有具身预训练的基线进行对比

## 相关概念

- [[Vision-Language-Action Model]]
- [[World Action Model]]
- [[LIBERO]]
