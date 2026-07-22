---
type: concept
aliases: [力觉记忆Token, 力觉Token, Force Memory, Wrench Memory Token]
---

# Force Memory Token

## 定义

将机器人腕部 6 轴力/力矩（wrench）的完整历史序列 $\{f_\tau\}_{\tau=1}^t$ 通过预训练 [[VAE]] 压缩为固定数量（$K$ 个）紧凑隐向量 token，并将其注入 VLA 动作专家作为长时接触事件记忆的条件信号。

## 数学形式

$$
\mathbf{T}_f = W_{\text{proj}} \cdot q_\phi(\hat{\mathbf{f}}_{1:t}) \in \mathbb{R}^{K \times d_{\text{model}}}
$$

其中 $q_\phi$ 为 [[Perceiver-IO]] 编码器（VAE 后验），$K=8$（FM-VLA 默认配置），$W_{\text{proj}}$ 为零初始化投影矩阵。

## 核心要点

1. **非马尔可夫感知**: 力觉 token 携带全局接触历史，使 VLA 能区分"已按 N 次"等视觉上无法分辨的状态
2. **紧凑编码**: 将任意长度（全集）的 wrench 序列映射到固定 $K$ 个 token，推理开销恒定（~3.3ms）
3. **编码器冻结**: Stage-1 预训练后冻结 VAE 编码器，Stage-2 仅微调 VLA 主干，避免遗忘
4. **零初始化注入**: 投影矩阵零初始化保证训练初期 force token 不破坏基座预训练分布
5. **宏观时序结构**: VAE 瓶颈自然过滤高频噪声，保留接触事件的宏观模式（对比 GRU 梯度消失、Q-Former 峰值过拟合）

## 代表工作

- [[FM-VLA]]: 提出 Force Memory Token 概念，通过 Masked ELBO + Free-Bits 训练的 VAE 实现

## 相关概念

- [[VAE]]
- [[Perceiver-IO]]
- [[Masked ELBO]]
- [[ForceVLA]]
- [[Pi0.5]]
