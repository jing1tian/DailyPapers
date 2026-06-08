---
type: concept
aliases: [Mean Per Joint Position Error, 平均关节位置误差]
---

# MPJPE (Mean Per Joint Position Error)

## 定义
人体姿态估计的标准评估指标，计算预测关节位置与真值关节位置的平均欧氏距离（单位：mm）。

## 数学形式
$$\text{MPJPE} = \frac{1}{J} \sum_{j=1}^{J} \| \hat{\mathbf{p}}_j - \mathbf{p}_j \|_2$$
其中 $J$ 为关节数，$\hat{\mathbf{p}}_j$ 为预测关节位置，$\mathbf{p}_j$ 为真值位置。

## 核心要点
1. 值越小越好，单位通常为毫米（mm）
2. PA-MPJPE（Procrustes Aligned）在刚性变换对齐后计算，排除全局旋转/平移误差
3. 常用于 3D 人体姿态估计、动作重建、HOI 重建评估

## 代表工作
- [[GRAIL]]: 用 MPJPE 评估 HOI 重建几何精度

## 相关概念
- [[HOI]]
- [[SMPL]]
