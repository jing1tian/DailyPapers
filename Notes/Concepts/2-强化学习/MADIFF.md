---
type: concept
aliases: [Multi-Agent Diffusion, 多智能体扩散RL]
---

# MADIFF

## 定义
将扩散模型应用于多智能体离线强化学习的框架，用扩散模型对联合轨迹分布建模，支持多智能体协调策略的学习。

## 数学形式
MADIFF 在多智能体轨迹空间定义去噪过程：
$$p(\tau^{1:N}) = \int p(\tau^{1:N}_0 | \tau^{1:N}_{1:T}) \prod_t p(\tau^{1:N}_t | \tau^{1:N}_{t-1})$$

## 核心要点
1. **联合轨迹建模**：同时对所有智能体的动作分布建模
2. **离线设置**：从离线数据集学习，无需在线交互
3. **安全缺口**：未考虑安全约束，被 CBF-Diffusion 等工作扩展

## 代表工作
- CBF-Guided Diffusion (2606.12640): 对比 MADIFF，加入 individual CBF 安全约束

## 相关概念
- [[SafeDiffuser]]
- [[MARL]]
- [[扩散模型]]
