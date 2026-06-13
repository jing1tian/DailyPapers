---
type: concept
aliases: [扩散强制, Diffusion Forcing]
---

# Diffusion Forcing

## 定义

Diffusion Forcing 是一种扩散模型训练范式，对序列中的每个时间步**独立采样**噪声水平 $\tau_t$（而非整个序列共享同一 $\tau$），使模型在训练时学会在任意混合噪声状态下预测，从而在自回归生成时维持长时序一致性。

## 数学形式

对于序列 $\{x^1_1, x^1_2, \ldots, x^1_T\}$，Diffusion Forcing 对每步独立采样：

$$
\tau_t \sim \mathcal{U}(0, 1), \quad x^{\tau_t}_t = \tau_t x^1_t + (1 - \tau_t) x^0_t
$$

训练目标（以[[流匹配]]为例）：

$$
\mathcal{L} = \mathbb{E}_{\{\tau_t\}} \left[ \sum_{t=1}^T \left\| (x^1_t - x^0_t) - f_\phi(x^{\tau_{1:t}}, \tau_{1:t}) \right\|_2^2 \right]
$$

## 核心要点

1. **独立噪声采样**：每时间步独立 $\tau_t$，模型学会处理不同步噪声配置
2. **长时序一致性**：由于模型在训练时见过部分已去噪/部分仍含噪的混合状态，推理时累积误差更小
3. **灵活推理调度**：可结合[[余弦噪声调度]]等策略在推理时按需分配去噪步数
4. **与 DDPM 的区别**：DDPM 中整个序列共享同一 $t$，Diffusion Forcing 每步独立

## 代表工作

- [[WEAVER]]: 在机器人操作世界模型中应用 Diffusion Forcing，显著改善 150 步（10s）长时序视频预测的 FID 指标
- Diffusion Forcing (Chen et al., 2024): 原始提出论文，在视频生成和决策任务上验证

## 相关概念

- [[流匹配]]
- [[余弦噪声调度]]
- [[扩散模型]]
- [[视频扩散模型]]
- [[Rectified Flow]]
