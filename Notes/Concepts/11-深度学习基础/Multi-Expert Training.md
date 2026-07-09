---
type: concept
aliases: [多专家协同训练, MECo, 多分支联合训练]
---

# Multi-Expert Training

## 定义

多专家协同训练（Multi-Expert Training）是一种训练范式，通过引入多个目标不同的专家网络（如视频预测专家、动作预测专家、几何重建专家）进行联合优化，训练时各专家共享表示并相互监督，推理时仅保留核心专家以维持部署效率。

## 数学形式

$$\mathcal{L}_{\text{total}} = \sum_{e \in \mathcal{E}} \lambda_e \mathcal{L}_e$$

其中 $\mathcal{E}$ 为所有专家集合，$\lambda_e$ 为各专家损失权重。推理时只使用核心专家子集 $\mathcal{E}_{\text{infer}} \subset \mathcal{E}$。

## 核心要点

1. **辅助专家蒸馏**: 仅训练时存在的辅助专家（如几何专家）将结构化知识隐式编码到共享表示中
2. **推理图不变**: 辅助专家在推理时完全移除，不增加延迟
3. **任务对齐监督**: 各专家的监督信号可设计为任务相关（如动作感知权重），避免无关噪声的干扰

## 代表工作

- [[MECo-WAM]]: 视频专家 + 动作专家 + 4D 几何专家三路协同训练，推理时移除 4D 分支
- [[GeoSem-WAM]]: 多辅助预测头（RGB/几何/语义）联合训练

## 相关概念

- [[Knowledge Distillation]]
- [[Mixed Attention]]
- [[World Action Model]]
- [[Flow Matching]]
