---
type: concept
aliases: [扩散去噪, 去噪扩散, DDPM, Score-based Generative Model]
---

# Denoising Diffusion

## 定义

去噪扩散（Denoising Diffusion）是一类生成模型范式：通过学习逐步逆转加噪过程来从噪声中恢复数据，包括 DDPM、DDIM、Flow Matching 等。在机器人领域，常用于从条件 KV Cache 中去噪生成动作序列。

## 数学形式

**前向加噪**（固定 Markov 链）：

$$q(x_s | x_0) = \mathcal{N}(x_s; \sqrt{\bar{\alpha}_s} x_0,\, (1-\bar{\alpha}_s) I)$$

**逆向去噪**（$S$ 步采样）：

$$x_0 \approx A_t = \text{Denoise}^{(S)}(\varepsilon,\, \mathcal{K}_t), \quad \varepsilon \sim \mathcal{N}(0, I)$$

其中 $\mathcal{K}_t$ 为条件 KV Cache，$S$ 为去噪步数。

## 核心要点

1. **条件生成**：以语言指令 + 视觉观测（编码为 KV Cache）为条件，去噪生成机器人动作
2. **步数与质量权衡**：步数 $S$ 越大，动作质量越高但推理越慢；VLA 中常用 $S=5\sim15$
3. **Flow Matching 变体**：π0.5 等模型使用 Flow Matching（ODE-based），而非传统 SDE
4. **与 KV Sharing 兼容**：[[OxyGen]] 中动作去噪和语言解码共享同一 prefill KV Cache

## 代表工作

- [[OxyGen]]: 在 MoT VLA 中以 $\mathcal{K}_t$ 为条件驱动动作去噪
- Diffusion Policy: 将扩散模型用于机器人操控策略学习
- [[π0.5]]: 使用 Flow Matching 的 VLA，以 KV Cache 为条件生成动作块

## 相关概念

- [[KV Cache]]
- [[Action Chunking]]
- [[Flow Matching]]
- [[Mixture-of-Transformers]]
