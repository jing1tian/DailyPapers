---
type: concept
aliases: [Huber Loss, 平滑L1损失, Smooth L1]
---

# SmoothL1 Loss

## 定义

SmoothL1 Loss（又称 Huber Loss）是一种结合 L1 和 L2 损失优点的回归损失函数：对小误差使用 L2（平滑、梯度稳定），对大误差使用 L1（对离群值不敏感）。

## 数学形式

$$
\text{SmoothL}_1(x) = \begin{cases} 0.5 x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}
$$

对于预测值 $\hat{y}$ 和真实值 $y$，整体损失为：

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \text{SmoothL}_1(\hat{y}_i - y_i)
$$

## 核心要点

1. **离群值鲁棒**: 当误差较大时退化为 L1，梯度有界（最大为 1），避免 L2 在离群值处的梯度爆炸
2. **小误差平滑**: 当误差较小时使用 L2，导数连续（在 $x=0$ 处可导），训练更稳定
3. **阈值可调**: 阈值参数（上式中为 1）可根据任务调整，控制 L1/L2 切换点
4. **广泛应用**: 目标检测的边界框回归（Faster R-CNN）、机器人操作的空间参数预测等

## 代表工作

- [[AffordanceVLA]]: 用于 How2Act 的 10 维空间布局参数回归（位置、朝向、包围盒）
- Faster R-CNN: 目标检测边界框回归的经典应用

## 相关概念

- [[扩散模型]]
- [[可供性（Affordance）]]
