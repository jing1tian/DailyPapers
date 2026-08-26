---
type: concept
aliases: [Q-guided sampling, Q guided diffusion, 值函数引导推理]
---

# Q-Guided Inference

## 定义
在扩散模型或 flow-matching 策略的去噪/推理过程中，用预训练 Q 函数的梯度引导采样方向，使生成动作朝高价值区域偏移，无需重训练策略网络。

## 数学形式
$$a_t \leftarrow a_t + \alpha \nabla_{a_t} Q(s, a_t)$$

在 flow-matching 中，引导信号在 ODE 积分路径上逐步施加。

## 核心要点
1. 推理时适配，不改变策略网络权重
2. Q 函数用离线 RL（IQL/IDQL）训练，提供 value 引导
3. 适用于需要场景特化但无法全量 fine-tune 的部署场景

## 代表工作
- [[QGF]]: Q-Guided Flow，用于 flow-matching VLA 的推理时适配

## 相关概念
- [[Flow Matching]]
- [[IQL]]
- [[Diffusion Policy]]
