---
type: concept
aliases: [TD Lambda, λ-回报, Lambda Return, 时序差分]
---

# TD(λ)

## 定义
TD(λ) 是一种结合多步时序差分估计的回报计算方法，通过参数 $\lambda \in [0,1]$ 在 Monte Carlo 回报（低偏差、高方差）和 TD(0)（高偏差、低方差）之间平滑插值。

## 数学形式

λ-回报递推定义：

$$
\mathbf{v}^k_t = r_t + \gamma\!\left((1-\lambda)V(s_{t+1}) + \lambda\,\mathbf{v}^k_{t+1}\right)
$$

边界条件：$\mathbf{v}^k_{t+K} = V(s_{t+K})$

价值网络损失：

$$
\mathcal{L}^{critic}(V) = \left\|V(s_t) - \mathbf{v}^k_t\right\|^2_2
$$

## 核心要点
1. $\lambda=0$：退化为单步 TD，偏差大但方差小
2. $\lambda=1$：退化为 Monte Carlo 回报，无偏但方差大
3. 典型设置 $\lambda=0.95$，$\gamma=0.995$，平衡两端
4. 可与模型辅助的想象轨迹结合，无需真实环境交互

## 代表工作
- [[WEAVER]]: 在潜在空间想象轨迹上用 λ-回报训练价值网络（λ=0.95, γ=0.995）
- Dreamer 系列: MBRL 中使用 TD(λ) 训练 Actor-Critic

## 相关概念
- [[Advantage Estimation]]
- [[世界模型]]
- [[流匹配]]
