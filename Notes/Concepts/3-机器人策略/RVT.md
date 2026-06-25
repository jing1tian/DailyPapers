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
- [[G3VLA]]: 将 RVT 列为"结构化 3D 表征"代表性工作之一，与其轻量级几何旁路注入思路对比——RVT 需要虚拟视图重渲染等任务专用设计，G3VLA 则直接复用预训练 VLA 的 2D 视觉 token 通路

## 相关概念
- [[VoxPoser]]
- [[PerAct]]: RVT 用虚拟视图渲染替代的体素化表征方法
