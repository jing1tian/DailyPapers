---
type: concept
aliases: [源感知残差适配器, Source-Aware Adapter, 源感知适配器]
---

# Source-Aware Residual Adapter

## 定义

一种轻量残差适配器设计，为不同来源（域）的数据分配独立的瓶颈投影参数，在共享主干网络上提供域特定的残差修正，从而在混合异质数据（如真实专家数据与生成伪数据）训练时降低域间干扰。

## 数学形式

$$
\operatorname{Ada}_d(x) = x + \alpha\, W_{\mathrm{up}}^d\, \sigma\!\left(W_{\mathrm{down}}^d\, \operatorname{LN}(x)\right)
$$

其中：
- $d \in \{\text{real}, \text{pseudo}\}$：数据来源域
- $W_{\mathrm{down}}^d, W_{\mathrm{up}}^d$：域专属瓶颈投影矩阵（$W_{\mathrm{up}}^d$ **零初始化**，保证初始为恒等映射）
- $\alpha$：残差缩放系数
- $\sigma$：非线性激活
- $\operatorname{LN}$：Layer Norm

## 核心要点

1. **共享骨干 + 域分支**: 主干动作网络对两域共享，适配器提供域特定修正，避免伪数据噪声污染真实数据的学习。
2. **零初始化**: $W_{\mathrm{up}}^d$ 全零初始化，训练初期适配器为恒等映射，稳定早期训练。
3. **推理时丢弃伪域**: 仅保留 `real` 域适配器用于部署，`pseudo` 域适配器在训练结束后丢弃。
4. **适用场景**: 知识蒸馏、合成数据增广、跨域迁移等需要混合不同质量数据来源的训练范式。

## 代表工作

- [[Vid2WAM]]: 首次将源感知残差适配器引入 WAM 训练，用于隔离真实专家动作与 IDM 伪动作的干扰

## 相关概念

- [[Knowledge Distillation]]
- [[Inverse Dynamics Model]]
- [[WAM]]
