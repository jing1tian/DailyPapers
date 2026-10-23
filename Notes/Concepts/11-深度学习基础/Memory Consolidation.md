---
type: concept
aliases: [记忆合并, 记忆压缩, memory merging, memory compression]
---

# Memory Consolidation

## 定义

当固定容量的记忆库（memory bank）发生溢出时，通过合并（而非随机删除）最相似的相邻条目来腾出空间，在保留记忆多样性的同时控制存储代价。

## 数学形式

$$
r_j^\star = \arg\max_r\, \cos\!\left(\mathrm{vec}(\mathbf{m}^j_r),\, \mathrm{vec}(\mathbf{m}^j_{r+1})\right)
$$

$$
\mathbf{m}^j_{r^\star} \leftarrow \frac{\mathbf{m}^j_{r^\star} + \mathbf{m}^j_{r^\star+1}}{2}
$$

找余弦相似度最高的相邻条目对，取均值合并，使记忆库减少一个条目。

## 核心要点

1. **相邻约束**：只合并时序上相邻的条目对（$r$ 与 $r+1$），保持时序连续性，避免将时间跨度大的状态混合。
2. **相似度优先**：优先合并最相似的相邻对，保留差异显著的状态（如抓取前后的关键状态）。
3. **均值合并**：合并方式为简单均值，保留两者的平均表示；对差异大的状态，合并会损失信息。
4. **与随机删除的区别**：随机删除可能丢失关键历史状态；合并策略通过相似性判断，倾向于保留状态变化大的时刻。

## 代表工作

- [[GeoMem-VLA]]: 在三路[[Causal Retrieval Memory|时序记忆]]中应用记忆合并策略

## 相关概念

- [[Causal Retrieval Memory]]: 记忆合并通常作为因果记忆检索的容量管理组件
- [[Temporal Context Buffer]]: 另一种时序上下文管理方式
