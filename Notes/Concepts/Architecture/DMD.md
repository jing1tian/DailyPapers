---
type: concept
aliases: [Distribution Matching Distillation, 分布匹配蒸馏]
---

# DMD

## 定义
Distribution Matching Distillation：一种扩散模型加速蒸馏方法，通过最小化学生模型输出分布与教师模型输出分布之间的散度来训练 few-step 生成模型。

## 数学形式
$$
\mathcal{L}_{\text{DM}} = D_{\text{KL}}\left(q_\phi(\mathbf{x}_0) \| p_\theta(\mathbf{x}_0)\right)
$$

实际实现通过梯度估计：
$$
\nabla_\phi \mathcal{L}_{\text{DM}} \approx \mathbb{E}\left[\nabla_\phi \log q_\phi(\mathbf{x}) \cdot \left(\log q_\phi(\mathbf{x}) - \log p_\theta(\mathbf{x})\right)\right]
$$

## 核心要点
1. 将 few-step 蒸馏问题转化为分布匹配，无需 step-by-step 的 trajectory 对齐
2. 比 Consistency Distillation 有更高的上限，但训练更复杂
3. DMD2 引入了双时间步 discriminator 进一步提升质量
4. [[CDM]] 将其推广到连续时间，解决离散时间步匹配误差累积问题

## 代表工作
- [[CDM]]: 连续时间版本的 DMD，理论上消除时间离散化误差
- [[MARBLE]]: 在 DMD 框架基础上做多奖励 RL fine-tuning
- [[HYWorld2]]: WorldStereo 2.0 使用 DMD 将多步扩散模型蒸馏为 4 步快速推理的学生模型

## 相关概念
- [[Diffusion Model]]
- [[Flow Matching]]
- [[CDM]]
