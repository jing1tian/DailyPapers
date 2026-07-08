---
type: concept
aliases: [Foresight Token, 可学习 Foresight Token, Latent Foresight]
---

# Learnable Foresight Token

## 定义

一种可学习的潜变量查询机制，通过与当前多模态上下文做交叉注意力，从冻结的视频生成模型中提炼任务相关的未来动态信息，以紧凑潜变量形式为机器人策略提供物理先验，推理时直接丢弃，无额外计算开销。

## 数学形式

$$
H^f = \text{CrossAttn}(Q^f, H_{\text{vlm}})
$$

$$
\mathcal{L}_{\text{video}} = \mathbb{E}\left\| \hat{F}_{t:t+4}(H^f) - F_{t:t+4} \right\|^2
$$

其中 $Q^f \in \mathbb{R}^{N_f \times d}$ 为可学习参数，$H^f$ 为 Foresight 条件嵌入，仅用于监督冻结视频生成模型，梯度只回传至 $Q^f$。

## 核心要点

1. **训练时有监督、推理时零开销**: 训练期间以冻结视频生成模型的未来帧预测为监督，推理时整个 Foresight 分支丢弃
2. **隐式物理先验提炼**: 无需在线生成像素级视频帧，将物理动态知识压缩到低维潜变量中
3. **对 OOD 泛化帮助显著**: 在 LIBERO-Plus 零样本任务上移除 Foresight Token 导致 -6.9 pt 下降

## 代表工作

- [[InternVLA-A1.5]]: 首次提出该机制，结合 MoT 架构与冻结 Wan 2.2 实现

## 相关概念

- [[World Model]]
- [[Flow Matching]]
- [[Wan 2.2]]
- [[Mixture-of-Transformers]]
- [[Action Chunking]]
