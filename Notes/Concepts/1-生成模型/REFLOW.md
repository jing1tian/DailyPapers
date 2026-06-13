---
type: concept
aliases: [ReFlow, Reflow Distillation, 反流蒸馏]
---

# REFLOW

## 定义
Flow Matching 的加速蒸馏方法，通过将 ODE 轨迹"拉直"（straightening）来减少推理所需的 NFE（函数评估次数），从而大幅加速生成速度。

## 数学形式
ReFlow 训练一个新的流场拟合已有 ODE 求解器轨迹：
$$v_\theta(x_t, t) \approx \frac{x_1 - x_0}{1}, \quad (x_0, x_1) \sim \text{ODE-trajectory}$$

## 核心要点
1. **轨迹拉直**：让向量场尽量平行，使得少量步骤即可到达终点
2. **蒸馏目的**：从多步 teacher 蒸馏到少步 student（甚至 1-step）
3. **与 [[CFM]] 的关系**：REFLOW 是 Flow Matching 上的二阶蒸馏操作

## 代表工作
- [[WEAVER]]: 用 REFLOW 对世界模型推理加速 5-10×
- Rectified Flow 原始论文

## 相关概念
- [[Rectified Flow]]
- [[CFM]]
- [[LCM]]
