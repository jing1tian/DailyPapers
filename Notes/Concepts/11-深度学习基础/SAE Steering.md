---
type: concept
aliases: [SAE 引导, Sparse Autoencoder Steering, 稀疏自编码器引导]
---

# SAE Steering (稀疏自编码器引导)

## 定义

通过训练稀疏自编码器（Sparse Autoencoder）将模型激活分解为可解释的稀疏特征方向，再对目标特征施加激活干预以引导模型行为。

## 数学形式

$$
h = W_{\text{dec}} \cdot \text{ReLU}(W_{\text{enc}} h + b_{\text{enc}}) + b_{\text{dec}}
$$

引导时将目标特征激活值置为指定强度。

## 核心要点

1. 属于 [[Activation Steering]] 的一种，但需要额外训练 SAE
2. 优点：特征语义可解释性强
3. 局限：SAE 训练成本高，特征对 VLA 动作专家的适配性有限
4. 对比 [[COAST]]：SAE 需要训练，COAST 仅需少量 rollout 即可构建 Conceptor

## 代表工作

- [[COAST]]: 将 SAE 引导作为基线，证明 Conceptor 方法在 VLA 上的优越性

## 相关概念

- [[Activation Steering]]
- [[Conceptor]]
- [[Representation Engineering]]
