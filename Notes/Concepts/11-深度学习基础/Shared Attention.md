---
type: concept
aliases: [共享注意力]
---

# Shared Attention（共享注意力）

## 定义

多个 Transformer 分支或模态共享注意力层的 Key（K）和 Value（V）矩阵，使得不同分支的 token 能够相互感知对方的信息，同时各分支保留独立的 Query（Q）以维持专门化能力。

## 核心要点

1. 跨分支信息流：Video DiT 和 Action DiT 的 token 在同一注意力层中互相可见
2. 参数效率：共享 KV 减少参数量，降低跨模态 cross-attention 的设计复杂度
3. 与 Cross-Attention 的区别：Shared Attention 是双向隐式共享，Cross-Attention 是单向显式条件化

## 代表工作

- [[Fast-WAM]]: Video DiT 与 Action DiT 通过 Shared Attention 实现信息交互
- [[GeoSem-WAM]]: 继承 Fast-WAM 的 Shared Attention 设计

## 相关概念

- [[Cross-Attention]]
- [[Self-Attention]]
- [[Mixture-of-Transformers]]
