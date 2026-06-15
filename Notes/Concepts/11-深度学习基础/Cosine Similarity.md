---
type: concept
aliases: [余弦相似度, 余弦距离, Cosine Distance]
---

# Cosine Similarity

## 定义

衡量两个向量方向相似程度的度量，取值范围 $[-1, 1]$，与向量模长无关，只关注方向。

## 数学形式

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2 + \varepsilon}
$$

作为损失函数时（最大化相似度）：

$$
\mathcal{L}_{\cos} = -\cos(\mathbf{x}, \mathbf{y})
$$

## 核心要点

1. **方向不变性**: 余弦相似度不受向量缩放影响，适合表征对齐场景（不同空间的特征维度不同）
2. **数值稳定**: 分母加 $\varepsilon$（如 1e-8）防止除零
3. **与 L2 的关系**: 对归一化向量，$\cos(\mathbf{x}, \mathbf{y}) = 1 - \frac{1}{2}\|\hat{\mathbf{x}} - \hat{\mathbf{y}}\|_2^2$（余弦距离等价于归一化 L2 距离）

## 代表工作

- [[AGRA]]: 用余弦相似度损失将视频 DiT 隐状态对齐至冻结 DINOv2 特征，解决 action-grounding gap

## 相关概念

- [[Representation Alignment]]
- [[DINOv2]]
- [[Contrastive Learning]]
