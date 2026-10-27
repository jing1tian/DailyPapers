---
type: concept
aliases: [键值记忆, KV Memory, Key-Value Store, 键值存储]
---

# Key-Value Memory（键值记忆）

## 定义

一种将信息编码为键（Key）-值（Value）对并通过查询（Query）检索的记忆机制；写入时将内容压缩为键值对存入存储，读取时用查询向量与键做相似度匹配，加权聚合对应值。

## 数学形式

写入投影：

$$
\bm{K} = \theta_K(\bm{h}), \quad \bm{V} = \theta_V(\bm{h})
$$

读取（软注意力）：

$$
\text{Mem}(\bm{q}) = \text{softmax}\!\left(\bm{q} \bm{K}^\top / \sqrt{d}\right)\bm{V}
$$

线性（快权重）形式：

$$
\text{Mem}(\bm{q}) = \bm{q}\, W, \quad W = V^\top K \left(K^\top K\right)^{-1}
$$

## 核心要点

1. **分离写/读路径**: 写入（人类/新场景数据）和读取（机器人/目标任务查询）可来自不同分布。
2. **参数记忆 vs 外部记忆**: 快权重是参数化键值记忆，Transformer 注意力是基于上下文的软键值记忆。
3. **跨分布桥接**: WAM-TTT 的 KVM 损失正是利用键值记忆对齐人类帧（写入）与机器人帧（读取）。

## 代表工作

- [[WAM-TTT]]: 键值记忆重建损失（KVM）——要求快权重将人类键映射到人类值，使机器人查询能读取人类语义

## 相关概念

- [[Fast Weights]]
- [[测试时训练]]
- [[注意力机制]]
- [[线性注意力]]
- [[Memory Module]]
