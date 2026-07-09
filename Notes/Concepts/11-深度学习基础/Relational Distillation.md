---
type: concept
aliases: [关系蒸馏, 关系知识蒸馏, Relational Knowledge Distillation, RKD]
---

# Relational Distillation

## 定义

关系蒸馏（Relational Distillation）是知识蒸馏的一种变体，以 token 或样本**之间的关系**（如 L2 距离、余弦相似度）而非绝对特征值为对齐目标，使学生网络继承教师网络的结构性几何关系，而不依赖共享的绝对特征坐标系。

## 数学形式

关系矩阵（L2 距离形式）：

$$R(i,j) = \|Z_i - Z_j\|_2$$

归一化关系蒸馏损失：

$$\mathcal{L}_{\text{rel}} = \frac{\sum_{i,j} w_{ij} \left|\hat{R}_{\text{student}}(i,j) - \hat{R}_{\text{teacher}}(i,j)\right|}{\sum_{i,j} w_{ij}}$$

其中 $\hat{R}$ 为归一化后的关系距离，$w_{ij}$ 为样本对权重。

## 核心要点

1. **坐标系无关**: 不要求学生网络与教师网络共享绝对特征空间，只对齐相对结构
2. **结构保持**: 保留教师的几何邻近性信息（哪些 token 在特征空间中更近）
3. **权重灵活**: 可引入任务相关权重（如动作感知权重）强调关键样本对

## 代表工作

- [[MECo-WAM]]: 以 [[VGGT]] 的 token 对 L2 距离为教师信号，通过动作感知权重加权对齐 4D 几何关系
- RKD（Relational Knowledge Distillation, CVPR 2019）: 开创性关系蒸馏工作

## 相关概念

- [[Knowledge Distillation]]
- [[VGGT]]
- [[Multi-Expert Training]]
- [[Temporal Geometry]]
