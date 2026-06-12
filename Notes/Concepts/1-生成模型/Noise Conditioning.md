---
type: concept
aliases: [噪声条件化, Noise Level Conditioning, τ Conditioning]
---

# Noise Conditioning

## 定义

扩散模型中将当前噪声水平 $\tau$（或时间步 $t$）作为条件输入注入网络，使模型感知当前去噪阶段，从而输出适当的去噪预测。

## 核心要点

1. **训练时**: 随机采样 $\tau \sim \mathcal{U}(0, 1)$，模型学习在不同噪声水平下去噪
2. **推理一致性**: 若推理时特征提取采用固定噪声水平，需与训练时保持分布一致（否则出现分布偏移）
3. **高噪声特征**: $\tau \approx 1$ 时特征保留全局语义和任务动态，适合动作条件化；低噪声时聚焦像素细节

## 与 AGRA 的关系

[[AGRA]] 在推理时将视频 DiT 固定在 $\tau_v^{\text{cond}} = 1$（高噪声）进行一次前向传播，提取用于动作控制的特征。这一设计确保训练-推理特征分布一致，是 AGRA 训练-推理一致性的核心设计点。

## 代表工作

- [[AGRA]]: 固定 $\tau_v^{\text{cond}}=1$ 的训练-推理一致性设计

## 相关概念

- [[Video Diffusion Model]]
- [[Cosmos-Predict-2.5]]
- [[World Action Model]]
