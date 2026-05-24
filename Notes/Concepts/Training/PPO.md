---
type: concept
aliases: [PPO, Proximal Policy Optimization, 近端策略优化]
---

# PPO (Proximal Policy Optimization)

## 定义
OpenAI 提出的近端策略优化算法，通过 clip 机制限制策略更新幅度，在计算效率和稳定性之间取得平衡，是目前最广泛使用的 on-policy RL 算法之一。

## 数学形式
$$
\mathcal{L}^\text{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 为策略比，$\hat{A}_t$ 为优势估计，$\epsilon$ 控制更新幅度。

## 核心要点
1. Clip 机制防止过大的策略更新（替代 TRPO 的二阶约束）
2. On-policy 算法，需要大量数据
3. 可与 value function baseline 结合减小方差
4. VLA RL fine-tuning 中常作为 baseline（被 GRPO 替代）

## 代表工作
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- [[SONIC]]：大规模人形机器人运动跟踪，PPO 训练主干算法

## 相关概念
- [[GRPO]]
- [[Reinforcement Learning]]
- [[DAPO]]
