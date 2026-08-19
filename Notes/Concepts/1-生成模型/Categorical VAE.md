---
type: concept
aliases: [离散VAE, Discrete VAE, 类别变分自编码器, 直通估计器VAE]
---

# Categorical VAE

## 定义

将潜变量建模为**离散类别分布**（而非连续高斯分布）的变分自编码器，使用 Straight-Through 估计器或 Gumbel-Softmax 技巧实现梯度反传。DreamerV3 的标准配置使用 32 个类别变量 × 32 个类别（共 1024-bit 潜编码）。

## 数学形式

$$
z \sim \text{Categorical}(p_1, p_2, \dots, p_K)
$$

$$
z_t \in \{0, 1\}^{C \times K},\quad \text{（32 类别 × 32 变量，独热编码）}
$$

**Straight-Through 梯度估计（前向离散，反向连续）:**

$$
\hat{z}_{ST} = z + (\text{softmax}(l) - \text{softmax}(l)).detach()
$$

## 核心要点

1. **为什么离散**: 离散潜空间对 world model 有正则化作用，防止潜变量空间崩塌（posterior collapse），DreamerV2/V3 实验表明比连续高斯潜空间效果更好。
2. **32×32 配置**: DreamerV3 使用 32 组独立的 32-类别 Categorical 分布，总潜变量维度 = 32×32 = 1024 bits（但用 32×32 = 1024 维 one-hot 向量表示）。
3. **Gumbel-Softmax vs Straight-Through**: 两者均可；DreamerV3 使用 Straight-Through 估计器（前向 argmax，反向 softmax 梯度）。
4. **与 VQ-VAE 的区别**: VQ-VAE 用码本查找（codebook lookup），Categorical VAE 用可学习的类别概率——前者码本固定大小，后者更灵活。

## 代表工作

- [[DreamerV3]]: 标准配置使用 32×32 Categorical VAE 作为 RSSM 的随机潜变量。
- [[QWM]]: 继承 DreamerV3 的 Categorical VAE 配置，形态条件化循环状态空间模型的一部分。

## 相关概念

- [[VAE]]
- [[VQ-VAE]]
- [[DreamerV3]]
- [[RSSM]]
