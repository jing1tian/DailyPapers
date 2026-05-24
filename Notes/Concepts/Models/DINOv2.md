---
type: concept
aliases: [DINOv2, DINOv2 ViT]
---

# DINOv2

## 定义
Meta AI 发布的自监督视觉特征提取器，使用蒸馏（DINO）和对比学习（iBOT）在大规模无标注数据上训练，输出强泛化性的密集视觉特征。

## 数学形式
$$
\mathcal{L} = \mathcal{L}_\text{DINO} + \lambda \mathcal{L}_\text{iBOT}
$$
联合优化全局蒸馏损失（DINO）和块级掩码预测损失（iBOT）。

## 核心要点
1. 无监督训练，无需标注，训练数据经过质量筛选（LVD-142M）
2. 输出的 patch 特征具有强空间语义，可直接用于分割、匹配
3. 常用作机器人感知、视频表征、世界模型的视觉 backbone
4. 比 ResNet、CLIP 的视觉特征更适合密集预测任务

## 代表工作
- Oquab et al. (2023): "DINOv2: Learning Robust Visual Features without Supervision"
- [[DiLA]]: 用 DINOv2 提取视频特征做解纠缠潜变量预测

## 相关概念
- [[DINO]]
- [[Vision Transformer]]
- [[CLIP]]
