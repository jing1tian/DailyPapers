---
type: concept
aliases: [AprilTag标定, AprilTag fiducial marker]
---

# AprilTag

## 定义

AprilTag 是一种专为机器人视觉设计的 2D 基准标记（fiducial marker）系统，通过黑白二值图案编码唯一 ID，可用于估计相机的 6-DoF 位姿。

## 核心要点

1. 每个 AprilTag 编码唯一整数 ID，允许同场景中放置多个标记
2. 仅需单目相机即可估计标记的完整 6-DoF 位姿（平移 + 旋转）
3. 对部分遮挡、光照变化和透视变形具有一定鲁棒性
4. 检测速度快，适合实时机器人应用
5. 常见系列：Tag16h5, Tag25h9, Tag36h11 等，数字越大编码容量越大

## 数学形式

通过已知标记物理尺寸 $s$ 和检测到的四个角点像素坐标 $\{p_i\}$，利用 PnP 问题求解相机位姿：

$$
\min_{R, t} \sum_{i=1}^{4} \| p_i - \pi(K [R | t] P_i) \|^2
$$

其中 $P_i$ 为标记角点的 3D 世界坐标，$K$ 为相机内参矩阵，$\pi$ 为透视投影。

## 代表工作

- [[VLA-REPLICA]]: 使用 AprilTag 实现跨实验室相机位姿标定和工作空间对齐
- [[SceneReplica]]: 使用视频叠加+AprilTag 确保抓取场景可复现
- [[RAMP]]: 基于 AprilTag 的组装任务标定
- [[FurnitureBench]]: 基于 AprilTag 的家具组装评测标定

## 相关概念

- [[Intel RealSense D455]]
- [[相机标定]]
