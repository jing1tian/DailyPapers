---
type: concept
aliases: [Advantage Weighted Actor-Critic]
---

# AWAC (Advantage Weighted Actor-Critic)

## 定义
一种离线到在线 RL 算法，通过用优势函数加权行为克隆损失实现从离线数据到在线探索的平滑过渡，避免 bootstrap 错误在策略外状态上的爆炸。

## 数学形式
$$\pi_{\theta} = \arg\max_\theta \mathbb{E}_{(s,a)\sim\beta}\left[\log\pi_\theta(a|s)\exp\left(\frac{1}{\lambda}A^\pi(s,a)\right)\right]$$

## 核心要点
1. 优势加权的 BC 损失：高优势动作得到高权重
2. 离线预训练 → 在线微调的统一框架，无需切换算法
3. 比纯 off-policy RL 更稳定，比纯 BC 更灵活

## 代表工作
- Nair et al. 2021, ICLR

## 相关概念
- [[SAC]]
- [[PPO]]
- [[Offline RL]]
