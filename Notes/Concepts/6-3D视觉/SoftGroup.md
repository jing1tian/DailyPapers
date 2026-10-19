---
type: concept
aliases: [SoftGroup segmentation]
---

# SoftGroup

## 定义
SoftGroup 是一种用于 3D 点云实例分割的方法，通过软分配（soft assignment）解决点云中物体边界模糊的问题，先做语义预测再做 bottom-up 实例聚合。

## 核心要点
1. 输入：3D 点云；输出：每个点的实例 ID + 语义类别
2. "Soft" 指每个点可软分配到多个实例候选，再通过 NMS 得到最终 instance mask
3. 常用于为 3D 场景 grounding 方法提供 instance proposals 预处理

## 代表工作
- [[Urbex]]：用 SoftGroup 提取 city-scale 场景 instance 信息（但最终推理为 candidate-free）
- [[CityAnchor]]：基于 SoftGroup proposals 做排序

## 相关概念
- [[CityRefer]]
- [[InstanceRefer]]
