---
type: concept
aliases: [Decoupled Weight Decomposition, Weight-Decomposed Low-Rank Adaptation]
---

# DoRA

## 定义
解耦权重分解低秩适应，将预训练权重分解为幅度（magnitude）和方向（direction）两个分量，对方向分量应用 LoRA，同时对幅度单独优化，兼顾 LoRA 的参数效率和全量微调的表达能力。

## 数学形式
$$W' = m \cdot \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c}, \quad \Delta W = BA$$

其中 $m$ 是可学习幅度向量，$\|\cdot\|_c$ 是列范数，$B, A$ 是低秩矩阵。

## 核心要点
1. 全量微调可分解为幅度变化 + 方向变化的组合
2. LoRA 只做方向调整（有偏差），DoRA 同时做幅度调整
3. 参数量仅比 LoRA 多幅度向量（远少于全量微调）
4. 在 VLA 后训练（CrossVLA）等场景优于标准 LoRA

## 代表工作
- [[DoRA]]：Liu et al. 2024，权重解耦低秩适应
- [[CrossVLA]]：跨范式 VLA 后训练中使用 DoRA

## 相关概念
- [[LoRA]]
- [[SFT]]
