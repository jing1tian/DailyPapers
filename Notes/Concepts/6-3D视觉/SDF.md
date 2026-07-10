---
type: concept
aliases: [Signed Distance Field, 有符号距离场, SDF表示]
---

# SDF

## 定义
一种隐式 3D 表示方法，空间中每个点存储其到最近表面的有符号距离（内部为负，外部为正，表面为零），可从中精确重建表面几何。

## 数学形式
$$\text{SDF}(\mathbf{x}) = s, \quad s \begin{cases} < 0 & \text{点在物体内部} \\ = 0 & \text{点在表面} \\ > 0 & \text{点在物体外部} \end{cases}$$

## 核心要点
1. 连续且可微，适合梯度优化（如 NeRF、物体重建）
2. 表面可用 Marching Cubes 等方法从等值面提取为 mesh
3. 对物理仿真有用：碰撞检测可直接查询 SDF 值判断穿透深度
4. 在 real-to-sim 场景重建中，用于精确重建物理关键交互区域

## 代表工作
- [[RoboSnap]]: 用 SDF 精确建模物理交互区域，保证仿真稳定性

## 相关概念
- [[3D Gaussian Splatting]]
- [[NeRF]]
