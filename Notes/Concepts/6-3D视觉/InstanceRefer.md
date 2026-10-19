---
type: concept
aliases: [InstanceRefer grounding]
---

# InstanceRefer

## 定义
InstanceRefer 是用于 3D 场景语言 grounding 的方法，基于 instance-level 特征进行语言-视觉对齐，直接对 3D instance proposals 做 referring expression comprehension。

## 核心要点
1. 以 3D instance proposals 为基本单元，而非逐点匹配
2. 在室内 3D grounding（ScanRefer 等）上建立了强 baseline
3. 作为 city-scale grounding 的 baseline 被扩展方法对比

## 相关概念
- [[CityRefer]]
- [[CityAnchor]]
- [[SoftGroup]]
