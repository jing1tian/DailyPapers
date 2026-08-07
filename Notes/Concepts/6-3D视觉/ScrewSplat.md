---
type: concept
aliases: [ScrewSplat, Screw Motion Gaussian Splatting]
---

# ScrewSplat

## 定义
ScrewSplat 是一种将 3D Gaussian Splatting 与螺旋运动（screw motion）理论结合的方法，用于从视频中重建带关节运动的物体，并拟合关节的旋转轴和平移参数。

## 核心要点
1. 螺旋运动（screw motion）统一描述旋转和平移，适合关节运动建模
2. 结合 3DGS 做外观建模 + 螺旋运动做关节参数估计
3. 可以从 RGB 视频直接恢复关节物体的运动参数，无需深度传感器
4. 输出可用于生成 [[URDF]] 文件，接入物理仿真器

## 代表工作
- [[RORA]]: 用 ScrewSplat 拟合关节运动，结合 3DGS 重建关节物体

## 相关概念
- [[3DGS]]
- [[URDF]]
- [[PartNet]]
- [[SplatSim]]
