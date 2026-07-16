---
type: concept
aliases: [隐空间匹配损失, 隐状态对齐损失]
---

# Latent Matching Loss

## 定义

在 ViT 隐空间中衡量预测状态与真实状态差异的组合损失，结合 L2 距离（约束量级）和余弦相似度（约束方向）。

## 数学形式

$$
\ell_{\text{lat}}(\hat{v}^l, v^l) = 0.1\,\|\hat{v}^l - v^l\|^2_2 + 0.9\!\left(1 - \frac{\langle \hat{v}^l, v^l \rangle}{\|\hat{v}^l\|_2\,\|v^l\|_2}\right)
$$

## 核心要点

1. 90% 的权重给余弦相似度，强调方向对齐（语义一致性）优于精确量级匹配
2. 10% 的 L2 距离防止解退化（两者都为零时余弦未定义）
3. 监督信号来自冻结的预训练 ViT（而非像素重建），计算高效

## 代表工作

- [[Orca]]: 使用此损失作为无意识学习和有意识学习的基础度量

## 相关概念

- [[Unconscious Learning]]
- [[Conscious Learning]]
- [[ViT]]
- [[Self-Supervised Learning]]
