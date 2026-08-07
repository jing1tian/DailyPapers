---
type: concept
aliases: [Unified Robot Description Format, 统一机器人描述格式]
---

# URDF

## 定义
URDF（Unified Robot Description Format）是 ROS 生态中用于描述机器人模型的 XML 格式文件，定义了机器人的运动学（links、joints）、动力学参数、碰撞体积和可视化模型。

## 核心要点
1. 描述机器人结构：link（刚体）和 joint（关节，支持 fixed/revolute/prismatic/continuous 类型）
2. 包含惯性参数（mass、inertia）用于物理仿真
3. 定义碰撞网格和视觉网格（通常用 `.dae` 或 `.stl` 格式）
4. 广泛用于 [[MuJoCo]]、[[IsaacLab]] 等仿真器的模型导入

## 数学形式
关节变换由 DH 参数或直接的 origin/rpy 定义：
$$T_{joint} = Trans(xyz) \cdot Rot(rpy)$$

## 代表工作
- [[RORA]]: 自动从 3DGS 重建中生成关节物体的 URDF

## 相关概念
- [[MuJoCo]]
- [[IsaacLab]]
- [[PartNet]]
