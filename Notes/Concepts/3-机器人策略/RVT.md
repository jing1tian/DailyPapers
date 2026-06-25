---
type: concept
aliases: [Robotic View Transformer]
---

# RVT

## 定义
多视角虚拟渲染 + Transformer 的机器人操作策略：将多视角 RGBD 重投影为统一虚拟视图，再用 Transformer 直接回归末端执行器位姿，避免体素化表征的高内存开销。

## 核心要点
1. 用虚拟相机重渲染多视角观测，替代 PerAct 等体素值图（voxel value map）表征
2. Transformer 直接在虚拟视图特征上预测关键帧动作，训练/推理更快
3. 常作为操作策略基线，与体素/点云类方法对比

## 代表工作
- 待补充

## 相关概念
- [[VoxPoser]]
