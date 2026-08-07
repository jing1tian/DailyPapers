---
type: concept
aliases: [Robotic View Transformer 2]
---

# RVT-2

## 定义
RVT-2 是 RVT（Robotic View Transformer）的升级版本，通过改进的多视角渲染和更高效的 transformer 架构，在 3D 机器人操控任务上取得更好的性能和效率。

## 核心要点
1. 基于 [[RVT]] 的多视角点云渲染方法，将 3D 场景投影为 2D 多视角图像
2. 使用更大容量的 VLM 骨架，提升指令跟随能力
3. 在 RLBench 等 3D 操控 benchmark 上达到 SOTA
4. 数据效率高，少量演示即可达到高成功率

## 代表工作
- [[BridgeVLA++]]: 将 RVT-2 作为 3D VLA baseline 进行对比

## 相关概念
- [[RVT]]
- [[SpatialVLA]]
- [[LOTUS]]
- [[PolarNet]]
