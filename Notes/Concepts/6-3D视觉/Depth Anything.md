---
type: concept
aliases: [Depth Anything V2, 深度估计基础模型]
---

# Depth Anything

## 定义

Depth Anything 是一个大规模预训练的单目深度估计基础模型，通过在海量未标注数据上进行自监督预训练，实现强泛化性的零样本深度估计能力。

## 核心要点

1. 基于 DINOv2 视觉编码器，大规模半监督训练
2. Depth Anything V2 进一步引入合成数据提升细粒度深度精度
3. 作为伪标签生成器，可为无深度传感器的场景提供深度估计标注
4. GeoSem-WAM 在展望未来工作时提到可用 Depth Anything 替代真实深度传感器标注

## 代表工作

- [[GeoSem-WAM]]: 建议未来使用 Depth Anything 等基础模型提供自监督几何伪标签

## 相关概念

- [[DINO]]
- [[Dense Prediction Transformer]]
- [[Dense Prediction]]
