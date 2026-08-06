---
type: concept
aliases: [MagicDrive Street View]
---

# MagicDrive

## 定义
MagicDrive：基于扩散模型的可控街景视频生成方法，通过 3D bounding box 和道路地图条件生成多相机一致的自动驾驶场景。

## 核心要点
1. 条件信号包括 3D box、道路布局、天气等
2. 多相机一致性约束保证几何正确
3. 用于自动驾驶数据增强和 world model 评测

## 代表工作
- [[DriftWorld]]: 引用 MagicDrive 的条件生成框架思路

## 相关概念
- [[VDM]]
- [[UniNav]]
