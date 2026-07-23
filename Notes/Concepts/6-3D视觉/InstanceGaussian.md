---
type: concept
aliases: [Instance-level Gaussian Splatting]
---

# InstanceGaussian

## 定义
在 3D Gaussian Splatting 基础上加入实例分割能力的方法，将场景中的每个物体实例与其对应的 Gaussian 集合关联。

## 核心要点
1. 扩展标准 3DGS：每个 Gaussian 附加实例 ID 属性
2. 支持实例级别的场景理解和操作（如 robot grasping 中的物体识别）
3. 是 ZeroSplat 等多目标指代分割方法的竞争基线之一

## 代表工作
- [[ZeroSplat]]: 作为 GR3DGS 基线对比方法之一

## 相关概念
- [[LangSplat]]（语言特征扩展的 3DGS）
- [[OpenGaussian]]（开放词汇 3DGS 理解）
- [[ReferSplat]]（单目标指代分割）
- [[ZeroSplat]]（多目标指代分割，今天的新工作）
