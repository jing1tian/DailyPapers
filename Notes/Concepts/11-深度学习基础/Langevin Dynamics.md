---
type: concept
aliases: [Langevin 动力学, Stochastic Gradient Langevin Dynamics, SGLD]
---

# Langevin Dynamics

## 定义
随机微分方程驱动的采样方法，在梯度下降步骤中注入适当幅度的高斯噪声，使样本在目标分布的支撑上遍历，常用于从能量函数或分数函数定义的分布中采样。

## 数学形式

连续形式：
$$
dx = \nabla_x \log p(x)\, dt + \sqrt{2}\, dW_t
$$

离散更新（扩散采样步）：
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta(x_t, t) \right) + \sigma_t \eta, \quad \eta \sim \mathcal{N}(0, \mathbf{I})
$$

## 核心要点
1. 扩散模型的采样过程本质上是 Langevin 动力学的离散近似
2. 分数函数 $\nabla_x \log p(x)$ 指引样本向高密度区域移动
3. CoME 将多专家的分数函数相加后代入 Langevin 更新，实现合成采样

## 代表工作
- [[CoME]]: 在 Langevin/DDPM 采样框架下合成多专家分数函数

## 相关概念
- [[扩散模型]]
- [[Product of Contrastive Experts]]
