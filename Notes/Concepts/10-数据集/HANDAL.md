---
type: concept
aliases: [Handheld Articulated and Labeled Dataset]
---

# HANDAL

## 定义
HANDAL：一个专注于手持铰接物体（articulated objects）的 6D 位姿估计数据集，包含真实场景下的 RGB-D 图像序列和精确位姿标注，用于评估类别级位姿估计的泛化能力。

## 数学形式
N/A（数据集）

## 核心要点
1. 包含工具类铰接物体（如剪刀、钳子等），难度高于刚体位姿估计
2. 真实场景采集，包含遮挡、背景杂乱等挑战
3. 被 F2G-Pose 用于验证在真实场景中的 camera-frame 几何预测迁移能力

## 代表工作
- [[F2G-Pose]]: 在 HANDAL 上测试 camera-frame 形状补全效果，验证 real-object 迁移能力

## 相关概念
- [[SOPE]]: 同类位姿估计评估数据集（合成）
- [[ROPE]]: 同类数据集
- [[FoundationPose]]: 相关基线方法
