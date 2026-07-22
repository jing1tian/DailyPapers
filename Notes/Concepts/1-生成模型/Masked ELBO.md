---
type: concept
aliases: [掩码ELBO, Masked Evidence Lower Bound, 带掩码的变分下界]
---

# Masked ELBO

## 定义

VAE 训练中的一种变体：重建损失仅计算被随机掩码选中的帧/位置子集 $\mathcal{M}$，而非所有输入位置，迫使隐变量学习整体语义结构而非记忆局部细节，类似 [[MAE]]（Masked Autoencoder）思想在生成模型中的应用。

## 数学形式

$$
\begin{aligned}
\mathcal{L}_{\text{masked-ELBO}} = &-\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x}_{1:T})} \left[\sum_{t \in \mathcal{M}} \log p_\theta(\mathbf{x}_t | \mathbf{z})\right] \\
&+ D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}|\mathbf{x}_{1:T}) \| p(\mathbf{z})\bigr)
\end{aligned}
$$

其中 $\mathcal{M} \subset \{1,\ldots,T\}$ 为随机选择的掩码帧集合。通常与 [[Free-Bits Regularization]] 结合使用。

## 核心要点

1. **防止过拟合**: 掩码重建迫使 VAE 学习跨帧的全局特征，避免隐变量退化为局部副本
2. **宏观结构学习**: 对力/力矩序列，掩码重建使模型关注接触事件的宏观时序模式，而非每帧精确幅值
3. **与 MAE 的区别**: MAE 是判别式自编码器，Masked ELBO 是生成式变分下界，保留了隐空间先验约束
4. **在 FM-VLA 中的使用**: 对 wrench 历史序列的随机帧掩码重建训练 Force-VAE，使 $K=8$ 个 token 捕获宏观接触模式

## 代表工作

- [[FM-VLA]]: 将 Masked ELBO 用于力觉序列 VAE 的预训练

## 相关概念

- [[VAE]]
- [[Free-Bits Regularization]]
- [[Perceiver-IO]]
- [[Force Memory Token]]
