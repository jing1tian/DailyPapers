---
type: concept
aliases: [Diffusion Forcing, 扩散强制, 扩散强制训练]
---

# Diffusion Forcing

## 定义
**Diffusion Forcing** 是 Chen et al. (2024) 提出的一种扩散模型训练框架，允许每个序列 token 有独立的噪声级别，统一了自回归生成与全序列扩散生成，特别适合用于视频预测和世界模型。

## 核心要点
1. 打破传统扩散模型"全序列同噪声级"的假设，每步可以独立设置噪声级
2. 训练时随机 mask 未来帧（用高噪声级模拟"未知"），实现自回归能力
3. 生成时可以在"确定性历史"和"随机未来"之间平滑插值
4. 与 Teacher Forcing 结合支持高效序列训练

## 数学形式
对序列 $x_1, x_2, \ldots, x_T$，每个 $x_t$ 的噪声级 $n_t$ 独立采样：
$$q(x_t^{n_t} | x_t) = \mathcal{N}(x_t^{n_t}; \alpha_{n_t} x_t, \sigma_{n_t}^2 I)$$

## 代表工作
- Chen et al. (2024): Diffusion Forcing: Next-token prediction meets full-sequence diffusion
- [[NanoWM]]: 基于 Diffusion Forcing 的极简世界模型实现

## 相关概念
- [[扩散策略]]
- [[世界模型]]
