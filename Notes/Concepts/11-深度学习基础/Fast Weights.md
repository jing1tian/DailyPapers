---
type: concept
aliases: [快权重, Fast-Weight Memory, TTT Fast Weights]
---

# Fast Weights（快权重）

## 定义

相对于模型主干"慢权重"（在大规模训练中缓慢更新）而言，快权重是一组轻量参数，可在极短时间内（几步 SGD 或单次前向）根据上下文信号快速更新，充当短期记忆或适应机制。

## 数学形式

典型 TTT 残差读出：

$$
\Delta\bm{z} = \theta_O \, f_W\!\left(\theta_Q(\bm{z})\right)
$$

快权重更新（内循环 SGD）：

$$
W_{i+1} = W_i - \eta\,\nabla_W \mathcal{L}_{\text{adapt}}(W_i)
$$

线性情形的闭式解（等价于 Hebbian 外积记忆）：

$$
W^* = V^\top K \left(K^\top K\right)^{-1}
$$

## 核心要点

1. **两级参数**: 慢权重（$\Theta$）在训练中更新，快权重（$W$）在推理/适应时更新。
2. **Hebbian 极限**: 线性快权重在等方性假设下等价于[[线性注意力|线性注意力]]的键值外积存储。
3. **容量限制**: 快权重 MLP 的大小决定了能编码多少上下文信息。
4. **元训练必要**: 快权重的有效性依赖元训练阶段学会"让快权重有用的初始化"，裸 TTT 无元训练效果差。

## 代表工作

- [[WAM-TTT]]: 在视频专家层插入快权重 MLP，通过 TTT 从人类视频中提取操作先验，残差叠加到视频流

## 相关概念

- [[测试时训练]]
- [[MAML]]
- [[Key-Value Memory]]
- [[线性注意力]]
- [[Memory Module]]
