---
type: concept
aliases: [奖励梯度, Reward Ascent, Differentiable Reward Gradient]
---

# Reward Gradient

## 定义

Reward Gradient（奖励梯度）：扩散模型对齐中，对 clean-output 预测空间中 local reward 函数的梯度，用于构建指向更高 reward 方向的显式改进目标。

## 数学形式

Local reward 定义为在 clean-output 解码后评分：

$$
\tilde{R}(y, c) = R(D(y), c)
$$

稳定化归一化 reward 梯度方向：

$$
u_{grad} = \frac{\nabla_y \tilde{R}(y, c)}{\|\nabla_y \tilde{R}(y, c)\|_2 + \epsilon_g}
$$

单步 reward 变化（Lemma B.4）：

$$
\tilde{R}(y + h_{step}\,u_{grad}) - \tilde{R}(y) \geq h_{step}\frac{\|g\|_2^2}{\|g\|_2 + \epsilon_g} - \frac{L\,h_{step}^2}{2}
$$

其中 $\epsilon_g$ 为梯度稳定器，$L$ 为局部 Lipschitz 常数。

## 核心要点

1. **局部有效性**：梯度方向保证在 trust-region 内的一阶 reward 改进，不宣称全局最优
2. **稳定化归一化**：除以 $(\|g\|_2 + \epsilon_g)$ 防止除零，同时作为隐式置信门控（梯度小时步长收缩）
3. **方向比半径更重要**：DiffusionOPSD 实验证明方向是目标构建效果的决定因素（CLIPScore 差距 0.08），远大于半径的影响
4. **与 ReFL 的关系**：ReFL 也通过 reward 梯度更新，但与模型优化耦合；DiffusionOPSD 将梯度用于构建 detached 目标

## 代表工作

- [[DiffusionOPSD]] (2608.24646)：将 reward 梯度用于 trust-region 内的显式正负目标构建
- [[ReFL]]：直接通过 reward 梯度反向传播更新模型

## 相关概念

- [[RLHF]]
- [[Trust Region]]
- [[DiffusionOPSD]]
- [[ReFL]]
- [[Behavior Policy]]
