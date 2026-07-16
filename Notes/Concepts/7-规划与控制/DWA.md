---
type: concept
aliases: [Dynamic Window Approach, 动态窗口法]
---

# DWA

## 定义
DWA（Dynamic Window Approach）是移动机器人局部避障规划算法，在速度空间中搜索可行速度，综合考虑障碍物距离、速度大小和方向偏差选取最优速度命令。

## 数学形式
$$
G(v, \omega) = \sigma\left(\alpha \cdot \text{heading}(v,\omega) + \beta \cdot \text{dist}(v,\omega) + \gamma \cdot v\right)
$$

在动态窗口约束（运动学约束 + 加速度约束）内最大化目标函数 $G$。

## 核心要点
1. **动态窗口**：由最大加速度限制在当前速度附近圈出可达速度集合
2. **实时性好**：仅在速度空间采样，计算量小，适合 ROS 部署
3. **ROS 标准组件**：`nav_stack` 默认局部规划器之一
4. **局限**：不能预见狭窄通道，高障碍密度下容易陷入局部最优

## 代表工作
- [[APPLV]]: 将 VLA 用于自适应调节 DWA 参数

## 相关概念
- [[TEB]]
- [[MPPI]]
- [[MPC]]
