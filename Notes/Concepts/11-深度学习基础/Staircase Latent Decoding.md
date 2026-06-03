---
type: concept
aliases: [Staircase Latent Decoding, 楼梯式潜空间解码, 楼梯式推理, Staircase CoT, 楼梯解码器]
---

# Staircase Latent Decoding

## 定义

楼梯式潜空间解码是一种将 Transformer 按中继深度（relay depth）$N_r$ 分区的并行连续链式推理机制：下层计算共享的视觉-语言基础表示，上层并行生成 $K_c$ 个连续潜推理状态，从而在单次前向传播中替代串行的自回归链式思考。

## 数学形式

楼梯解码器生成 $K_c$ 个连续潜状态：

$$
\hat{y}_{1:K_c} = \mathcal{F}_{\text{stair}}(x; N_r)
$$

冻结语言模型（Qwen3.5-0.8B）从潜状态重建文本 CoT 用于监督：

$$
P_\phi(r_{1:M_r} \mid \mathbf{z}_{1:K_c})
$$

重建损失（仅优化推理分支和前缀投影头）：

$$
\mathcal{L}_{\text{CoT}} = -\sum_{m=1}^{M_r} \log P_\phi(r_m \mid \mathbf{z}_{1:K_c}, r_{<m})
$$

## 核心要点

1. 将 Transformer 深度在 $N_r$ 处切分：下层（共享中继层）产生视觉-语义联合基础，上层并行分支各自产生一个潜推理状态
2. $K_c$ 个潜状态在**一次前向传播**中并行生成，相比串行自回归 CoT 大幅减少推理延迟
3. 通过冻结小型 LM 的文本重建监督（frozen text reconstruction）防止潜状态坍塌为无意义向量
4. 只有推理分支参数和前缀投影头（$P_\text{pref}$）参与训练，VLM 骨干全程冻结
5. 与统一模式推理（Unified Mode）配合：楼梯解码生成事件结构化的潜推理，指导后续固定长度块的动作预测

## 代表工作

- [[WALL-WM]]: 提出 Staircase Latent Decoding，在统一推理模式中以并行潜 CoT 代替串行语言推理

## 相关概念

- [[大语言模型]]
- [[视觉语言动作模型]]
- [[交叉注意力]]
