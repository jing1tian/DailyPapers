---
type: concept
aliases: [动作等价未来瓶颈, Future Bottleneck, Quotient Bottleneck, 特权信息蒸馏]
---

# Action-Equivalent Future Bottleneck（动作等价未来瓶颈）

## 定义

一种 Teacher-Student 蒸馏结构，Teacher 在训练时可见当前和未来视觉摘要（特权信息），Student 仅可见当前信息；通过动作重建、嵌入蒸馏和几何保持三项损失约束，使 Student 在无未来帧时仍能准确预测动作。

## 数学形式

Teacher 和 Student 嵌入：

$$
z_t = q_t([c, f, s_0]), \quad z_s = q_s([c, s_0])
$$

联合损失：

$$
\mathcal{L}_q = \lambda_{\text{act}}^q \mathcal{L}_{\text{rec}}^q + \lambda_{\text{dist}}^q \|z_s - \text{sg}(z_t)\|_2^2 + \lambda_{\text{geom}}^q \mathcal{L}_{\text{geom}}
$$

其中几何保持损失要求 batch 内样本对在嵌入空间的距离与动作空间距离一致（"等价"之名的由来）。

## 核心要点

1. **特权信息蒸馏**：训练时 Teacher 观察未来，部署时只保留因果 Student 路径
2. **几何保持**：通过 $\mathcal{L}_{\text{geom}}$ 使嵌入空间保留动作结构，避免单纯拉近距离损失结构信息
3. **"等价"含义**：压缩后的嵌入 $z_s$ 应与动作空间等距（isometric），而非与 Teacher 嵌入完全一致
4. **部署因果性**：移除 Teacher 路径后，Student 路径严格不依赖未来帧

## 代表工作

- [[DECOWAM]]: 引入此瓶颈结构，在移动操作中将 A-MSE 降低 3.8%（消融），同时保持推理因果性

## 相关概念

- [[知识蒸馏|Knowledge Distillation]]
- [[梯度反转层|Gradient Reversal Layer]]
- [[Residual Adapter]]
