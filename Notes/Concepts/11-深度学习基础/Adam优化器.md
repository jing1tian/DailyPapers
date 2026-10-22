---
type: concept
aliases: [Adam, Adam Optimizer, 自适应矩估计]
---

# Adam 优化器

## 定义

Adam（Adaptive Moment Estimation）是一种自适应学习率优化算法，结合动量（一阶矩）和自适应学习率（二阶矩）对梯度进行校正，是深度学习中最常用的优化器之一。

## 数学形式

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

## 核心要点

1. **一阶矩估计**（动量）：平滑梯度方向，减少震荡
2. **二阶矩估计**（自适应学习率）：对频繁更新的参数降低有效学习率
3. **偏差修正**：$\hat{m}_t, \hat{v}_t$ 修正早期步骤的偏差
4. **默认超参数**：$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$

## 代表工作

- [[TAPDreamer]]: 用于离线对抗补丁优化，500步迭代

## 相关概念

- [[SGD]]
- [[AdamW]]
