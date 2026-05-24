---
type: concept
aliases: [Contrastive Activation Addition, 对比激活加法]
---

# CAA (Contrastive Activation Addition)

## 定义

通过从对比提示对（正向/负向）提取的激活差值向量，在推理时以加法方式干预 Transformer 的中间层激活，引导模型行为。

## 数学形式

$$
\vec{v} = \frac{1}{N}\sum_{i=1}^{N}(h^+_i - h^-_i)
$$

$$
h'_\ell = h_\ell + \alpha \cdot \vec{v}
$$

其中 $h^+, h^-$ 分别为正向和负向提示在层 $\ell$ 的激活，$\alpha$ 为引导强度。

## 核心要点

1. 由 Turner et al. (2024) 提出，属于 [[Activation Steering]] 的加法变体
2. 简单高效，无需额外训练
3. 局限：仅使用秩一方向，未充分利用子空间结构
4. 对比 [[COAST]]：CAA 用加法向量，COAST 用 Conceptor 乘法门控

## 代表工作

- [[COAST]]: 将 CAA 作为基线进行对比，证明 Conceptor 子空间建模的优越性

## 相关概念

- [[Activation Steering]]
- [[Conceptor]]
- [[Representation Engineering]]
