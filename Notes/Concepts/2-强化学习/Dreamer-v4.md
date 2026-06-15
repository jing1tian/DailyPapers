---
type: concept
aliases: [DreamerV4, Dreamer v4]
---

# Dreamer-v4

## 定义
Dreamer-v4 是 Dreamer 系列基于模型的强化学习方法的最新版本，通过在潜在空间学习世界模型并在想象轨迹上训练 Actor-Critic 来解决稀疏奖励和长时序规划问题。

## 数学形式

世界模型组件：
- **RSSM**（循环状态空间模型）：$h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$
- **Actor**：$a_t \sim \pi_\theta(\cdot | h_t, z_t)$
- **Critic**：$V_\phi(h_t, z_t) \approx \mathbb{E}[G_t]$

## 核心要点
1. 从头学习图像编码器，可能损害预训练模型的分布外鲁棒性
2. RSSM 结构将确定性隐状态和随机隐状态解耦
3. 通过在想象轨迹上的反向传播训练 Actor（梦境学习）

## 代表工作
- [[WEAVER]]: 指出 Dreamer-v4 从头学习编码器的 OOD 鲁棒性问题，使用预训练 VAE 改进

## 相关概念
- [[世界模型]]
- [[VAE]]
- [[TD(λ)]]
