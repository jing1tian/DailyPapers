---
type: concept
aliases: [Smooth L1损失, Smooth L1 Loss, Huber Loss, Huber损失]
---

# Smooth L1 损失（Huber 损失）

## 定义
Smooth L1 损失（又称 Huber 损失）在误差较小时行为类似 L2 损失（平方），误差较大时行为类似 L1 损失（线性），结合两者优点，对异常值具有鲁棒性。

## 数学形式

$$
\text{SmoothL1}(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}
$$

## 核心要点
1. 小误差区间平滑可导（梯度稳定），大误差区间线性增长（抗异常值）
2. 常用于目标检测的边界框回归和机器人位姿估计
3. 相比 MSE 对标注噪声和离群点更鲁棒

## 代表工作
- [[AffordanceVLA]]: How2Act 的 10-DoF 位姿回归使用 Smooth L1 损失

## 相关概念
- [[MSE损失]]
- [[二值交叉熵]]
- [[可供性预测]]
