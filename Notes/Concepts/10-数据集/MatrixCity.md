---
type: concept
aliases: [Matrix City Dataset]
---

# MatrixCity

## 定义
大规模城市级 NeRF/3DGS 训练与评测数据集，包含用游戏引擎渲染的高分辨率城市街景，覆盖不同光照条件和相机轨迹。

## 核心要点
1. 基于虚幻引擎渲染，提供 ground truth depth 和相机位姿
2. 城市级尺度（数平方公里），专为大规模场景重建设计
3. 包含 aerial（无人机视角）和 street（地面视角）两种拍摄模式
4. 是 TideGS 等大规模 3DGS 系统的标准 benchmark

## 代表工作
- [[TideGS]]（2026）: 十亿级 3DGS 训练，MatrixCity 上测试 out-of-core 优化

## 相关概念
- [[3D Gaussian Splatting]]
- [[NeRF]]
- [[TideGS]]
