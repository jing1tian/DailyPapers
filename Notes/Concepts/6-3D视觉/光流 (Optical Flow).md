---
type: concept
aliases: [Optical Flow, 光流, 光学流]
---

# 光流 (Optical Flow)

## 定义

光流（Optical Flow）是图像序列中像素在时间维度上的运动向量场，描述视频帧之间各像素点的位移估计，是计算机视觉中运动分析的核心工具。

## 数学形式

光流约束方程（亮度恒定假设）：

$$
\frac{\partial I}{\partial x} u + \frac{\partial I}{\partial y} v + \frac{\partial I}{\partial t} = 0
$$

其中 $(u, v)$ 为像素水平/垂直方向速度。

## 核心要点

1. **密集光流 vs 稀疏光流**: 密集光流估计所有像素；稀疏光流（如 Lucas-Kanade）仅跟踪特征点
2. **HOF（光流直方图）**: 将光流方向和幅度统计为方向直方图，是轻量动作特征的经典表达
3. **机器人运动感知**: 在操作任务中用于感知机械臂运动幅度，与夹爪接触事件密切相关

## 代表工作

- [[SKIP]]: SKIP-Selector 视觉特征流使用 36 维 HOF 特征捕捉机械臂运动

## 相关概念

- [[FILM]]（依赖光流进行帧插值）
- [[场景流]]
