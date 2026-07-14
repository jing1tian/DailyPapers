---
type: concept
aliases: [噪声策略, noise policy, latent noise policy]
---

# Noise Policy

## 定义

Noise Policy 是一个轻量的状态条件函数 $\pi^w(s)$，用于替代生成策略采样阶段的随机噪声 $w \sim \mathcal{N}(0,I)$，在不修改冻结基础模型权重的前提下引导生成输出。

## 数学形式

$$
a = \pi_{gp}\!\left(s,\, \pi^w(s)\right)
$$

训练目标（MSE 回归损失）：

$$
\mathcal{L}(\phi) = \mathbb{E}_{(s,\, w^*) \sim \mathcal{D}} \left\| \pi^w(s) - w^* \right\|_2^2
$$

其中 $w^*$ 由 [[Action Inversion]] 从专家纠正动作逆映射得到。

## 核心要点

1. **冻结基础模型**: 噪声策略参数 $\phi$ 单独优化，基础策略 $\pi_{gp}$ 完全冻结，避免先验遗忘。
2. **结构简单**: 观测编码器 + MLP，参数量远小于基础模型（~1–10M vs 数十亿）。
3. **先验保留机制**: 由于 $\pi_{gp}$ 固定，噪声策略只能将输出引导至基础模型的表达范围内，几何上约束适应行为。
4. **双缓冲区训练**: 结合干预样本（逆映射噪声）和成功 rollout 样本，防止过拟合稀疏干预。

## 代表工作

- [[FlowDAgger]]: 提出 Noise Policy 范式用于在线人机交互适应

## 相关概念

- [[Action Inversion]]
- [[流匹配]]
- [[扩散策略]]
- [[DAgger]]
- [[行为克隆]]
