---
type: concept
aliases: [Action Diffusion Transformer, 动作扩散变换器]
---

# Action-DiT

## 定义

在 [[World Action Model]] 框架中专门负责建模机器人动作分布的 Diffusion Transformer 专家，与视觉专家（[[Video-DiT]]）通过 [[Joint Attention|联合注意力]] 交换信息，以 [[Flow Matching|流匹配]] 目标训练动作去噪。

## 数学形式

后训练阶段注意力（读取视觉 token，单向）：

$$
H_A = \operatorname{Attn}(Q_A,\; [K_V;\, K_A],\; [V_V;\, V_A])
$$

推理时动作去噪 ODE：

$$
\frac{da_{\tau_a}}{d\tau_a} = v^a_{\theta_A}(a_{\tau_a},\, \tau_a \mid \mathcal{C}_V,\, l,\, q),\quad \tau_a: 1 \to 0
$$

## 核心要点

1. **与 Video-DiT 解耦**：Action-DiT 有独立参数，可在 Video-DiT 冻结时单独训练
2. **动作专属流匹配**：以动作块 $a_{t:t+H-1}$ 为建模对象，噪声时间步 $\tau_a$ 独立于视频流
3. **推理高效**：配合 [[Asymmetric Attention|非对称注意力]] 和 [[KV-Cache|视觉缓存]]，推理时无需运行 Video-DiT

## 代表工作

- [[NAVA-WAM]]: 提出在无标注视频上直接预训练 Action-DiT，实现原生动作先验学习

## 相关概念

- [[Video-DiT]]
- [[Mixture-of-Transformers]]
- [[Flow Matching]]
- [[Joint Attention]]
- [[Asymmetric Attention]]
- [[Action Chunking]]
