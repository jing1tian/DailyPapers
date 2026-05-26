---
type: concept
aliases: [Recurrent State Space Model, 循环状态空间模型]
---

# RSSM

## 定义
Recurrent State Space Model，一种在潜空间中建模序列动态的循环模型，核心思想是将状态分解为确定性部分（由 GRU 捕获）和随机部分（由 VAE 捕获），用于视频预测和 model-based RL。

## 数学形式
$$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$$
$$z_t \sim q_\phi(z_t | h_t, o_t) \quad \text{(posterior)}$$
$$\hat{z}_t \sim p_\phi(\hat{z}_t | h_t) \quad \text{(prior)}$$

## 核心要点
1. 确定性隐状态 $h_t$ 由 GRU 更新，捕获长期依赖
2. 随机隐状态 $z_t$ 从后验（给定观测）或先验（预测）采样
3. 训练目标包含重建损失和 KL 散度（ELBO）
4. 可在潜空间中 rollout 多步，用于 MPC 和 imagination-based RL

## 代表工作
- [[DreamerV3]]: 最成熟的 RSSM 实现，加入 symlog 变换和动态范围正则化
- [[Dreamer]]: 原始 Dreamer 论文，引入 RSSM 用于连续控制

## 相关概念
- [[DreamerV3]]
- [[世界模型]]
- [[扩散世界模型]]
- [[Model Predictive Control]]
