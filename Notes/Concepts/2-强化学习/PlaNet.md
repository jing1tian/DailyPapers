---
type: concept
aliases: [Deep Planning Network, Planet]
---

# PlaNet

## 定义
DeepMind 提出的基于学习世界模型的规划算法，在紧凑隐空间（RSSM）中学习动态模型，然后用 CEM 做纯隐空间规划（无需训练策略网络）。

## 数学形式
RSSM（Recurrent State Space Model）：
$$h_t = f_\phi(h_{t-1}, s_{t-1}, a_{t-1})$$
$$s_t \sim p_\phi(s_t \mid h_t), \quad \hat{o}_t \sim p_\phi(\hat{o}_t \mid h_t, s_t)$$

CEM 规划：在 latent rollout 中优化 action sequence。

## 核心要点
1. 首个在 latent space 做 model-based planning 的端到端可扩展方法
2. [[Dreamer]] 系列（DreamerV2、DreamerV3）在其基础上加了 actor-critic
3. 被 FlowMo-WM、LeWorldModel 等工作引用为经典 WM baseline
4. RSSM 成为此后众多 world model 的标准架构

## 代表工作
- Hafner et al. (2019) 原始论文
- [[FlowMo-WM]]：在 PlaNet/RSSM 框架上扩展 ambient drift 建模

## 相关概念
- [[Dreamer]]
- [[DreamerV3]]
- [[RSSM]]
- [[Action-Conditioned World Model]]
