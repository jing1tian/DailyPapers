---
type: concept
aliases: [Diffusion NFT, DiffusionNFT Framework]
---

# DiffusionNFT

## 定义
一种将扩散模型与 RL fine-tuning 结合的训练框架，将扩散去噪过程建模为 Markov 决策过程，使用策略梯度方法直接优化人类偏好奖励。

## 数学形式
$$
\mathcal{L}_{\text{DiffusionNFT}} = \mathbb{E}_{\mathbf{x}_0 \sim p_\theta}\left[r(\mathbf{x}_0)\right] - \beta \cdot D_{\text{KL}}(p_\theta \| p_{\text{ref}})
$$

将每个去噪步骤视为一个动作，整条去噪轨迹视为一个 episode。

## 核心要点
1. 把扩散模型的去噪过程 $\mathbf{x}_T \to \mathbf{x}_0$ 看作 sequential decision-making
2. 用 REINFORCE/PPO 风格的梯度更新直接优化奖励函数
3. [[MARBLE]] 在此框架基础上解决多奖励梯度冲突问题
4. 相比基于偏好对比的 DPO，可以使用任意可微或黑盒奖励

## 代表工作
- [[MARBLE]]: 在 DiffusionNFT 框架上提出多奖励梯度调和方法
- [[DiffusionART]]: 诊断 DiffusionNFT 不稳定根源（一致性力 + EMA 复原机制），提出聚合优先引导彻底消除一致性力

## 相关概念
- [[Diffusion Model]]
- [[MARBLE]]
- [[FlowGRPO]]
- [[GRPO]]
