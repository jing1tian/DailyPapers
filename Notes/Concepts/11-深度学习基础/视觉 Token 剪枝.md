---
type: concept
aliases: [Visual Token Pruning, token pruning, 视觉token压缩, 视觉token剪枝]
---

# 视觉 Token 剪枝

## 定义

在 [[Transformer]] 模型推理过程中，通过评估视觉 token 的重要性，动态删除冗余或低重要性的视觉 token，以降低计算量和推理延迟，同时尽量保持模型输出质量。

## 数学形式

$$
\min_{f} \mathcal{L}(\mathcal{P}, \tilde{\mathcal{P}}) \quad \text{s.t.} \quad |f(\mathbf{E}_v)| = \tilde{M}
$$

其中 $f$ 为 token 选择函数，$\tilde{M} < M$ 为目标保留数量，$\mathcal{P}$ 和 $\tilde{\mathcal{P}}$ 分别为原始和剪枝后的输出分布。

## 核心要点

1. **重要性评估**：通常基于注意力分数（attention score）评估每个视觉 token 对输出的贡献，重要性低的 token 被删除。
2. **VLM vs VLA 差异**：在 [[视觉语言动作模型|VLA]] 中，prefill 阶段和 action-decode 阶段的注意力分布显著不同，仅使用 prefill 注意力会导致动作关键 token 被误删。
3. **无训练 vs 微调**：无训练（training-free）方法直接利用推理时注意力，微调方法则通过端到端训练学习稀疏注意力模式。
4. **压缩收益**：视觉 token 数量通常远多于文本 token（每帧 256+ token），剪枝后可获得显著的 FLOPs 和延迟降低。

## 代表工作

- [[FastV]]: 基于语义注意力的 VLM token 剪枝，适用于图文理解但不适合 VLA
- [[SparseVLM]]: 文本-视觉交叉注意力驱动的 token 稀疏化
- [[DivPrune]]: 基于多样性的视觉 token 选择
- [[VLA-Pruner]]: 联合语义+动作双重注意力的 VLA 专用 token 剪枝，无训练，最高 1.99× 加速

## 相关概念

- [[Self-Attention]]
- [[Transformer]]
- [[视觉语言动作模型]]
- [[最大最小多样性问题]]
- [[指数移动平均]]
