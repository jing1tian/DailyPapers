---
type: concept
aliases: [MoT, Mixture of Transformers]
---

# Mixture-of-Transformers (MoT)

## 定义

Mixture-of-Transformers 是一种多专家 Transformer 架构，不同专家分支（如 Video DiT 和 Action DiT）共享注意力层的 Key/Value，实现跨模态/跨任务的信息交互，同时保持各分支的专门化能力。

## 核心要点

1. 共享注意力（Shared Attention）：多个分支在 Transformer 层中共享 Key 和 Value，降低参数量的同时实现跨分支信息流
2. 与 Mixture-of-Experts (MoE) 的区别：MoE 是在 FFN 层做稀疏激活，MoT 是在注意力层做显式多分支共享
3. 推理时可选择性激活分支（如关闭 Video DiT 分支）以提升效率
4. World Action Model 中用于将视觉世界表示传递给动作生成分支

## 代表工作

- [[Fast-WAM]]: 采用 MoT 架构的高效 WAM 基线
- [[GeoSem-WAM]]: 在 Fast-WAM MoT 架构上增加几何语义辅助监督
- [[Cosmos3]]: NVIDIA 在 Cosmos 3 中将 MoT 扩展为双流架构（AR 推理子序列 + DM 生成子序列），通过 joint attention 实现跨模态物理推理与生成的统一
- [[AffordanceVLA]]: 使用含理解、可供性生成、动作三专家的 MoT 架构，通过 UAA 单向注意力机制协调，在机器人操作 VLA 中实现感知-动作解耦

## 相关概念

- [[Mixture-of-Experts]]
- [[Shared Attention]]
- [[Video Diffusion Transformer]]
- [[World Action Model]]
