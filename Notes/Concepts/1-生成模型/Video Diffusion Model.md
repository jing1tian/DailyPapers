---
type: concept
aliases: [视频扩散模型, VDM]
---

# Video Diffusion Model

## 定义

Video Diffusion Model 是将扩散模型（Diffusion Model）扩展到视频生成领域的方法，通过时空去噪过程生成时序一致的视频序列。

## 数学形式

$$
p_\theta(\mathbf{x}_{0:T_v}) = \prod_{t=1}^{T_v} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

其中 $\mathbf{x}_{0:T_v}$ 为视频帧序列，去噪过程在时空维度上联合建模。

## 核心要点

1. **时序一致性**: 在帧间建模时序依赖关系，保证生成视频的运动连贯性
2. **时空注意力**: 使用 3D 或因果时空注意力机制捕获跨帧关系
3. **潜在扩散**: 通常在 VAE 压缩的潜在空间中操作以降低计算复杂度
4. **噪声时刻条件化（[[Noise Conditioning]]）**: 通过去噪时刻 $\tau$ 条件化模型

## 代表工作

- [[AGRA]]: 使用 Cosmos-Predict-2.5-2B 作为视频扩散骨干，通过 AGRA 对齐中间特征提升机器人动作预测

## 相关概念

- [[Video Diffusion Transformer]]
- [[World Action Model]]
- [[DINOv2]]
- [[Representation Alignment]]
