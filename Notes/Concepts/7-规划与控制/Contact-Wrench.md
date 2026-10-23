---
type: concept
aliases: [接触力旋量, Contact Wrench, Wrench, Force-Torque]
---

# Contact Wrench

## 定义
接触力旋量（Contact Wrench）是描述机器人末端执行器与环境接触时完整力学状态的六维向量，包含三维力（force）和三维力矩（torque）。

## 数学形式
$$\mathbf{w} = \begin{bmatrix} \mathbf{f} \\ \boldsymbol{\tau} \end{bmatrix} \in \mathbb{R}^6$$

其中 $\mathbf{f} \in \mathbb{R}^3$ 为接触力，$\boldsymbol{\tau} \in \mathbb{R}^3$ 为接触力矩。

## 核心要点
1. 比单独的力或力矩更完整地描述接触状态
2. 是混合运动-力控制（Hybrid Motion-Force Control）的基础变量
3. 对力敏感操作（插拔、装配、柔性物体操作）至关重要
4. 通常由腕部六轴力传感器测量

## 代表工作
- [[CW-WAM]]: 首个将接触力旋量纳入世界动作模型预测目标的框架
- [[ForceVLA]]: 力感知 VLA 策略
- [[FoAR]]: 基于力感知的动作表示

## 相关概念
- [[Hybrid Motion-Force Control]]
- [[Impedance Control]]
