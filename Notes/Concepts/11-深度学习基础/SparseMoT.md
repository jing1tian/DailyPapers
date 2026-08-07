---
type: concept
aliases: [Sparse Mixture-of-Tokens, 稀疏token混合]
---

# SparseMoT (Sparse Mixture-of-Tokens)

## 定义
SparseMoT 是一种稀疏 KV cache 注入机制，将外部表示（如 world model 的 future representation）稀疏地注入到 transformer 的 KV 序列中，只携带最关键的 token，在保留语义信息的同时显著降低推理计算量。

## 核心要点
1. 不同于全量 KV 融合，SparseMoT 只选取信息密度高的 token 注入
2. 适用于 World Action Model 推理时将 future representation 条件化到 action model
3. 推理时 KV 序列长度显著缩短，降低 attention 计算复杂度
4. 训练时仍使用完整 future representation，只在 inference 时稀疏化

## 代表工作
- [[SparseMoT-WAM]]: 首次将 SparseMoT 用于 WAM 推理效率优化

## 相关概念
- [[WAM]]
- [[MoT]]
- [[KV Cache]]
