---
type: concept
aliases: [无意识学习, 隐式学习]
---

# Unconscious Learning

## 定义

通过无标注连续视频数据，让模型在隐空间中学习密集的自然状态转移，模拟人类婴儿期通过感知流（无语言）获取物理世界直觉的过程。

## 数学形式

$$
S_{t+\Delta} \sim p_\Theta^u(S_{t+\Delta} \mid S_t,\, z_t)
$$

损失函数：

$$
\mathcal{L}_{\text{obs}} = \mathbb{E}\left[\ell_{\text{lat}}\left(\hat{v}^l_{t+1},\, v^l_{t+1}\right)\right]
$$

## 核心要点

1. 无需语言标注，利用视频的时序连续性作为自然监督
2. 学习密集状态转移（每一帧都是预测目标）
3. 与[[Conscious Learning|有意识学习]]互补：前者覆盖低层物理动态，后者覆盖高层语义事件

## 代表工作

- [[Orca]]: 将无意识学习定义为视频自监督状态转移预测，通过可学习查询 $Q_1$ 实现

## 相关概念

- [[Conscious Learning]]
- [[Next-State Prediction]]
- [[Self-Supervised Learning]]
- [[Latent Matching Loss]]
