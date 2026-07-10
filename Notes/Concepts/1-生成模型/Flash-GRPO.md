---
type: concept
aliases: [Flash GRPO, 单步随机 GRPO]
---

# Flash-GRPO

## 定义
针对视频扩散模型的高效 GRPO 强化学习后训练方法，通过"单一随机步骤"策略解决 Flow-GRPO 中的信用分配问题，配合 Coefficients-Preserving Sampling (CPS) 和时步平衡梯度重加权，实现精确且稳定的策略更新。

## 核心要点
1. **单步随机探索**：每个 GRPO 组内只在一个随机选定的去噪步骤处引入随机性，其余步骤保持确定性 ODE；优势估计在同一时步进行比较，实现精确信用分配
2. **CPS 采样**：使用 [[CPS|Coefficients-Preserving Sampling]] 替代 SDE 转化，保持边际分布与确定性采样路径一致，避免 SDE 引入的过量噪声和梯度不稳定
3. **时步平衡重加权**：对策略梯度按转移增益的倒数重加权 $\lambda_k \propto \kappa_k^{-1}$，防止少数高增益时步主导训练
4. **无 KL 惩罚**：不使用参考模型或 KL 正则，严格在线更新，每批样本对应单次梯度更新

## 数学形式

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\lambda_k \hat{A}^{(i)} \log \pi_\theta\left(x_{t_{k+1}}^{(i)} \mid x_{t_k}^{(i)}\right)\right]
$$

其中 $\lambda_k$ 为时步平衡权重，$\hat{A}^{(i)}$ 为组内标准化优势。

## 代表工作
- Flash-GRPO (He et al., arXiv 2605.15980)
- [[LingBot-Video]]: 采用 Flash-GRPO 范式进行多维奖励后训练

## 相关概念
- [[GRPO]]
- [[FlowGRPO]]
- [[CPS]]
- [[Rectified Flow]]
