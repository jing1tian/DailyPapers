---
type: concept
aliases: [Flow GRPO, Flow Matching GRPO]
---

# FlowGRPO

## 定义
将 GRPO 策略优化算法应用于 Flow Matching 生成模型的训练框架，通过组内相对奖励直接优化 flow-based 生成质量。

## 数学形式
$$
\mathcal{L}_{\text{FlowGRPO}} = \mathbb{E}_{t, \mathbf{x}_t}\left[\sum_{i=1}^{G} \hat{A}_i \cdot \log p_\theta(\mathbf{x}_0^{(i)} | \mathbf{x}_t)\right]
$$

在 flow trajectory 的随机时间步上采样 $G$ 个输出，组内做 advantage 估计。

## 核心要点
1. 将 GRPO 从离散 token 序列推广到连续 flow trajectory
2. 在 text-to-image 和视频生成上比 DPO 类方法更灵活
3. [[MARBLE]] 将其作为对比基线，指出 FlowGRPO 处理多奖励时存在梯度冲突

## 代表工作
- [[MARBLE]]: FlowGRPO 的对比基线
- [[DiffusionART]]: 将 FlowGRPO 作为基线，在 HunyuanVideo-13B 上以 11.72× 更少 GPU 时间超越其晚期奖励水平

## 相关概念
- [[Flow Matching]]
- [[GRPO]]
- [[DiffusionNFT]]
- [[Diffusion Model]]
