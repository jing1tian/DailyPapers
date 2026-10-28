---
type: concept
aliases: [Conformal Quantile Regression, Conformal Prediction]
---

# CQR

## 定义
CQR（Conformal Quantile Regression）：基于 conformal prediction 框架的分位数回归方法，用于为模型预测提供统计覆盖保证的不确定度区间，无需强分布假设。

## 数学形式
对校准集 $\{(x_i, y_i)\}_{i=1}^n$，计算 non-conformity score：

$$s_i = \max(q_\alpha(x_i) - y_i,\ y_i - q_{1-\alpha}(x_i))$$

预测区间：$C(x) = [q_\alpha(x) - \hat{q}, q_{1-\alpha}(x) + \hat{q}]$，其中 $\hat{q}$ 为校准分位数。

## 核心要点
1. 提供 marginal coverage 保证（$(1-\alpha)$ 的样本落在区间内）
2. 不依赖模型分布假设，适用于任意黑盒预测器
3. 在 robotics 中可用于 VLA action 的不确定度校准

## 代表工作
- [[SentryVLA]]: 用 CQR 校准 VLA 的动作不确定度，驱动 confidence-aware action selection

## 相关概念
- [[SentryVLA]]
