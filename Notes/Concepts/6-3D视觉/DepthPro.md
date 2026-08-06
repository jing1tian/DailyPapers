---
type: concept
aliases: [Apple Depth Pro]
---

# DepthPro

## 定义
DepthPro：Apple 发布的单目深度估计模型，可在无需相机内参的情况下输出度量尺度（metric scale）深度图。

## 核心要点
1. 输出绝对尺度深度，而非相对深度
2. 无需相机内参，直接预测焦距
3. 高分辨率输入，适合精细场景重建

## 代表工作
- [[InfiniSplat]]: 用 DepthPro 提取深度先验辅助单图 3DGS 生成

## 相关概念
- [[InfiniSplat]]
- [[DINOv2]]
