---
type: concept
aliases: [KITTI 360, KITTI360]
---

# KITTI-360

## 定义
大规模自动驾驶数据集，覆盖约 73.7 km 的 360° 环境扫描，包含激光雷达、鱼眼相机和语义标注，常用于新视图合成和 3DGS 评测。

## 核心要点
1. 73.7 km 驾驶场景，含稠密 3D 语义标注
2. 多传感器：激光雷达 + 鱼眼 + 透视相机
3. 常用于评测 NeRF/3DGS 在大规模室外场景的重建质量

## 代表工作
- [[DepthSplat]]：在 KITTI-360 上做立体视图 3DGS
- [[StereoSplat+]]：单视角立体 3DGS 的 KITTI-360 评测

## 相关概念
- [[3D Gaussian Splatting]]
- [[NeRF]]
