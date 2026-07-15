---
type: concept
aliases: [时序上下文缓冲区, 历史动作队列, Context Buffer, ω]
---

# Temporal Context Buffer

## 定义

**时序上下文缓冲区**（Temporal Context Buffer）是一种利用**历史动作 token 的紧凑队列**（而非存储原始观测帧）来为当前决策提供时序上下文的机制，专门用于解决多阶段任务中的[[Perceptual Aliasing|感知混叠]]问题。

## 数学形式

$$
\omega_{t-H':t} = \{a_{t-H'}, a_{t-H'+1}, \ldots, a_{t-1}\}
$$

增强后的策略目标：

$$
\max_\theta \mathbb{E}\bigl[\log \pi_\theta(\varphi, a_{t:t+H} \mid o_t, \ell, \omega_{t-H':t})\bigr]
$$

## 核心要点

1. **紧凑性**: 存储历史 action token（离散码），而非原始观测帧，效率更高
2. **阶段感知**: 不同阶段的历史动作序列不同，即使当前帧视觉相同，上下文也能区分任务阶段
3. **训练正则**: Stage 3 训练中以 0.5 的 dropout 率随机丢弃历史动作，增强策略的鲁棒性
4. **典型场景**: 倒水任务（倒水前后视觉相似）、擦白板（擦除进度跟踪）等多阶段操作任务

## 代表工作

- [[Lumo-2]]: 通过时序上下文缓冲区解决多阶段任务中的感知混叠，历史 action token dropout=0.5

## 相关概念

- [[Perceptual Aliasing]]: 时序上下文缓冲区所解决的核心问题
- [[Action Chunking]]: 历史 action chunk 构成上下文缓冲区的内容
- [[Latent World Dynamics]]: 与时序上下文协同，进一步丰富状态表征
