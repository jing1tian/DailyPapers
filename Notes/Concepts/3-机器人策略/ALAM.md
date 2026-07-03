---
type: concept
aliases: [Additive-and-Reversal Latent Action Module, Latent Action Module]
---

# ALAM

## 定义

ALAM（Additive-and-Reversal Latent Action Module）是一种自监督潜在动作提取模块，从相邻视频帧对中学习帧级运动意图向量，通过加法一致性和反转一致性约束保证潜在动作空间的几何连贯性，无需动作标注即可从纯视频数据中训练。

## 数学形式

提取：

$$
m_t = E_m(I_t, I_{t+1}) \in \mathbb{R}^{d_m}
$$

加法一致性：

$$
\mathcal{L}_{add} = \left\|m_i^k - (m_i^j + m_j^k)\right\|_2^2
$$

反转一致性：

$$
\mathcal{L}_{rev} = \left\|m_i^j + m_j^i\right\|_2^2
$$

总损失：

$$
\mathcal{L}_{LAM} = \lambda_{vq}\mathcal{L}_{vq} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{add}\mathcal{L}_{add} + \lambda_{rev}\mathcal{L}_{rev}
$$

## 核心要点

1. 冻结编码器 $E_m$ 从帧对 $(I_t, I_{t+1})$ 提取紧凑潜在动作向量 $m_t$
2. 加法一致性：从帧 $i$ 到帧 $k$ 的潜在动作 = 经过中间帧 $j$ 的两步之和
3. 反转一致性：正向与反向动作互为相反数
4. 训练时可利用无标注的纯视频数据，扩大可用训练数据规模
5. 作为 ABot-M0.5 中时间粒度对齐的关键桥接表示

## 代表工作

- [[ABot-M0.5]]: 将 ALAM 提取的 $m_t$ 作为中间桥接表示，解决 WAM 的时间粒度失配问题

## 相关概念

- [[Latent-Action]]
- [[LAM]]
- [[CFM]]
- [[Hierarchical Latent Action]]
- [[ABot-M0.5]]
