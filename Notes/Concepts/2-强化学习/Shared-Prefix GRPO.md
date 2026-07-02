---
type: concept
aliases: [共享前缀 GRPO, SharedPrefix GRPO]
---

# Shared-Prefix GRPO

## 定义

Shared-Prefix GRPO 是对 [[GRPO]] 的改进：在同一组 rollout 中，强制所有成员共享相同的前缀轨迹（从同一仿真器状态出发），只对分叉点之后的 suffix 轨迹计算组相对优势，从而消除接近阶段差异对奖励比较的干扰。

## 核心要点

1. **动机**: 机器人操作任务通常分为接近阶段 + 精细操作阶段；标准 [[GRPO]] 中同组 rollout 的接近质量不同，会把接近行为的随机性引入组相对比较，稀释精细操作的学习信号
2. **实现**: leader rollout 执行前缀至分叉点，克隆仿真器状态初始化其余成员；各成员独立执行 suffix，suffix 回报用于组相对比较
3. **掩码策略**: 前缀动作块从策略梯度损失中排除（只有 suffix 块进入 GRPO 目标），避免将不同 suffix 的回报分配给完全相同的前缀动作
4. **适用场景**: SFT 后接近行为已可靠，精细交互仍需 RL 改进的任务；追求训练稳定性时优于 [[Tree-Structured Prefix Branching]]

## 数学形式

前缀块 $(x, a) \notin \mathcal{B}_i$（掩码），只有 suffix 块参与：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{\sum_{i} |\mathcal{B}_i|} \sum_{i} \sum_{(x,a) \in \mathcal{B}_i} \min\!\left(\rho_\theta \cdot A_i,\ \text{clip}(\rho_\theta, 1{-}\eta, 1{+}\eta) \cdot A_i\right)
$$

其中 $\mathcal{B}_i$ 仅含 suffix 段动作块。

## 代表工作

- [[Z-1]]: 提出并应用于 flow-based VLA 模型的 [[GRPO]] 后训练

## 相关概念

- [[GRPO]]
- [[Tree-Structured Prefix Branching]]
- [[Action Chunking]]
