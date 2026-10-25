---
type: concept
aliases: [Synthetic Object Pose Estimation dataset]
---

# SOPE

## 定义
SOPE：用于类别级 6D 物体位姿估计的合成数据集，提供精确的位姿、尺寸和形状标注，是 F2G-Pose 的主要训练和评估 benchmark。

## 数学形式
N/A（数据集）

## 核心要点
1. 合成生成，标注精确，用于训练类别级位姿估计模型
2. 配套 ROPE（真实场景版本）用于 sim-to-real 评估
3. F2G-Pose 仅在 SOPE 合成数据上训练，在 ROPE 和 HANDAL 上做迁移评估

## 代表工作
- [[F2G-Pose]]: 主要训练集，与 ROPE 组成 sim-to-real 评估对

## 相关概念
- [[HANDAL]]: 真实场景铰接物体数据集
- [[FoundationPose]]: 相关基线
