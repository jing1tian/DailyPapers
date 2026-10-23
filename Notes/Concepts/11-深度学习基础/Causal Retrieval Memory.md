---
type: concept
aliases: [因果记忆检索, Causal Retrieval, 时序因果检索, Geometry-Aware Episode Memory]
---

# Causal Retrieval Memory

## 定义

一种用于序贯决策（如机器人操作）的时序记忆检索机制，核心约束是**仅允许检索当前时刻之前（同一 episode 内）的记忆条目**，确保因果性，防止信息泄露。通常配合[[Memory Consolidation|记忆合并]]策略管理有限容量的记忆库。

## 数学形式

$$
\widetilde{\mathbf{x}}^j_t = R_j\!\left(\mathbf{x}^j_t,\, \mathcal{M}^j_{<t};\, t\right)
$$

$\mathcal{M}^j_{<t}$：第 $j$ 路在时刻 $t$ 之前的全部记忆条目；$t$ 作为时间戳编码注入检索注意力，使模型感知时序关系。

## 核心要点

1. **因果性约束**：记忆库仅包含 $t$ 之前的条目，每个 episode 边界时重置，避免跨集污染。
2. **多路并行**：典型实现包含视觉流（×2）和几何流（×1）三路独立记忆，各自维护独立记忆库。
3. **时间戳编码**：将绝对/相对时间戳注入跨注意力的 key/value，帮助模型识别记忆的时序顺序。
4. **门控融合**：检索到的几何记忆与视觉记忆通过软门控（sigmoid）融合，自适应控制几何信息注入比例。
5. **容量管理**：记忆库容量有限，溢出时触发[[Memory Consolidation|记忆合并]]，将最相似的相邻条目均值合并。

## 代表工作

- [[GeoMem-VLA]]: 将因果记忆检索与几何监督（[[GeoPlan Token]]）结合，用于 [[VLA]]

## 相关概念

- [[Memory Consolidation]]: 记忆库溢出时的合并策略
- [[GeoPlan Token]]: 几何流的记忆条目来源
- [[MemoryVLA]]: 另一种 VLA 时序记忆方法
- [[Temporal Context Buffer]]: 类似的时序上下文缓冲机制
