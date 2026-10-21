---
type: concept
aliases: [Huber Loss, Huber 函数, smooth L1 loss, SmoothL1]
---

# Huber 损失

## 定义

Huber 损失（Huber Loss）是介于 L1 损失和 L2 损失之间的鲁棒损失函数：在误差较小时使用 L2（对小误差更敏感），在误差较大时使用 L1（对离群点更鲁棒）。

## 数学形式

$$
\text{Huber}(x; \delta) = \begin{cases}
\frac{1}{2}x^2 & \text{if } |x| \leq \delta \\
\delta\left(|x| - \frac{1}{2}\delta\right) & \text{otherwise}
\end{cases}
$$

其中 $\delta$ 为切换阈值（超参数）。梯度为：

$$
\frac{\partial}{\partial x}\text{Huber}(x;\delta) = \begin{cases}
x & \text{if } |x| \leq \delta \\
\delta \cdot \text{sign}(x) & \text{otherwise}
\end{cases}
$$

## 核心要点

1. **对离群点鲁棒**: 大误差时梯度有界（上界为 $\delta$），避免梯度爆炸
2. **连续可微**: 在 $x = \pm\delta$ 处连续且一阶导连续
3. **阈值选择**: $\delta$ 越大越接近 L2，$\delta \to 0$ 越接近 L1
4. **等价 smooth L1**: PyTorch `nn.SmoothL1Loss` 即 Huber 损失的变体（$\delta=1$）

## 代表工作

- [[EVA]]: 用 Huber 惩罚关节加加速度（jerk）和加速度，作为视频世界模型可执行性奖励的平滑性惩罚项

## 相关概念

- [[可执行性奖励]]
- [[逆动力学模型]]
