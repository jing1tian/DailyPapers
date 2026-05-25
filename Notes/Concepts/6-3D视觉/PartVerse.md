---
type: concept
aliases: [PartVerse Dataset]
---

# PartVerse

## 定义
部件级（part-level）3D 物体数据集/框架，为关节体和复杂物体提供精细的部件分割标注，支持物理仿真资产生成。

## 核心要点
1. 提供部件级别的 3D 语义分割，超越整体物体层面的标注
2. 支持关节体的部件分解（如门板、把手、铰链等）
3. 被 [[PhysX-Omni]] 用作关节体生成的数据/方法基础
4. 与 [[PartNet]] 类似但专注于仿真就绪的物理属性标注

## 代表工作
- [[PhysX-Omni]]: 使用 PartVerse 数据支持关节体部件分解

## 相关概念
- [[PartNet]]: 部件标注 3D 数据集的先驱工作
- [[Articulate-Anything]]: 关节体生成的相关方法
- [[TRELLIS]]: 3D 几何生成骨干网络
