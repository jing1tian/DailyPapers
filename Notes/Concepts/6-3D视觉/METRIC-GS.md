---
type: concept
aliases: [Metric-Scale 3D Gaussian Splatting]
---

# METRIC-GS

## 定义
从 mono-depth 先验中选择性估计 metric scale 的 3DGS 方法：用 MLE 框架只在统计置信度高的区域应用 depth 对齐，避免 naive quantile 对齐的不稳定性。

## 核心要点
1. 3DGS 从 SfM 继承任意单位，不能直接用于 AR 或测量
2. MLE 选择性策略：只在 depth prior 置信度高的区域施加 scale 约束
3. 使用 [[MoGe]] 作为 mono-depth prior
4. 在 [[ScanNet]], [[TUM]] 室内场景上评估

## 相关概念
- [[3DGS]]
- [[MoGe]]
- [[ScanNet]]
