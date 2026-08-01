---
type: concept
aliases: [Querying Transformer, Q-Former, 查询 Transformer, 过渡级Q-Former, Transition-level Q-Former]
---

# Q-Former

## 定义

一种轻量级 Transformer 模块，通过一组可学习查询向量（Learnable Query Tokens）在固定规模的输出空间内提取输入表示的精华信息，常用于视觉-语言对齐或特征压缩。

## 数学形式

$$
\mathbf{q} = \text{QFormer}(\mathbf{Q};\, \mathbf{x}_1, \mathbf{x}_2)
$$

其中 $\mathbf{Q} \in \mathbb{R}^{M \times D}$ 为 $M$ 个可学习查询向量，通过交叉注意力联合读取输入特征 $\mathbf{x}_1, \mathbf{x}_2$，输出固定长度的特征序列 $\mathbf{q} \in \mathbb{R}^{M \times D}$。

## 核心要点

1. **固定输出长度**: 无论输入序列多长，Q-Former 始终输出 $M$ 个 token，实现可控压缩
2. **可学习查询**: 查询向量在训练中学习提取任务相关信息，充当信息瓶颈
3. **跨模态/跨帧注意力**: 可联合注意多个输入源（图像、时序帧对等），灵活适配不同任务
4. **过渡级变体（Transition-level Q-Former）**: 对相邻视频帧对 $(\mathbf{x}^i, \mathbf{x}^{i+1})$ 分别提取转移特征，引入局部时序归纳偏置

## 代表工作

- BLIP-2 (Li et al., 2023): 提出 Q-Former 用于视觉-语言预训练中的特征对齐
- [[PhiZero]]: 将 Q-Former 扩展为"过渡级 Q-Former"，提取视频帧对间的物理状态转移特征

## 相关概念

- [[Physical Language]]: 过渡级 Q-Former 的输出经 FSQ 离散化后得到物理语言
- [[有限标量量化|FSQ]]: Q-Former 输出的离散化方案
- [[Attention Pooling]]: 类似的特征聚合思路
