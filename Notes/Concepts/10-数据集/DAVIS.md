---
type: concept
aliases: [DAVIS Dataset, Densely Annotated VIdeo Segmentation]
---

# DAVIS

## 定义
视频目标分割（video object segmentation）领域的标准评测基准，提供高质量、像素级标注的真实世界短视频片段，常被动态场景重建、视频追踪、4D 重建等任务用作评测集或前景/遮挡掩码来源。

## 核心要点
1. 视频分辨率高、标注精细，是衡量分割/追踪算法在真实视频上泛化能力的常用测试集
2. 在动态 3D/4D 重建工作中常被用来评测 in-the-wild 视频上的重建质量，或为重建管线提供前景掩码
3. 不含相机标定或深度真值，本身不是 3D 重建数据集，只提供 2D 分割标注

## 代表工作
- [[Lift4D]]: 用 DAVIS 视频做 in-the-wild 单目 4D 重建的定性/定量评测

## 相关概念
- [[SAM]]
- [[CoTracker3]]
