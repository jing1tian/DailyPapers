---
type: concept
aliases: [GenPose, Generative Pose Estimation]
---

# GenPose

## 定义
基于生成模型的类别级物体位姿估计方法，通过学习物体形状先验来恢复 3D 旋转和平移，无需逐实例 CAD 模型。

## 核心要点
1. 使用生成模型（通常为扩散模型或 flow-based）建模位姿后验分布
2. 依赖类别级形状先验（shape prior）来处理形状内的变化
3. 输入 RGB(-D) 图像，输出位姿分布或点估计
4. 不需要参考图像，也不需要实例级 CAD 模型

## 代表工作
- [[F2G-Pose]]: 在 GenPose 基础上引入几何感知 Foundation Feature，去掉了形状先验依赖

## 相关概念
- [[FoundationPose]]
- [[RADIO]]
- [[RegDiff]]
