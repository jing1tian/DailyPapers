---
type: concept
aliases: [Sparse Token Dropping, Token Pruning, 稀疏Token剪枝]
---

# SPRINT

## 定义
SPRINT（Sparse Progressive Inference with Token Dropping）是一种在 Transformer 训练和推理中随机丢弃 patch token 的加速方法，通过减少参与注意力计算的 token 数量来降低计算复杂度，同时保持生成质量。

## 数学形式

在每个 Transformer 层，以概率 $p$ 随机 mask patch token：

$$
\tilde{X} = X \odot \mathbf{m}, \quad \mathbf{m}_i \sim \text{Bernoulli}(1-p)
$$

仅对未被 mask 的 token 计算注意力，计算复杂度从 $O(N^2)$ 降低到 $O((N(1-p))^2)$。

## 核心要点
1. Drop 概率 $p=0.5$ 时，注意力计算量减少约 75%（$N^2$ → $0.25N^2$）
2. 训练时随机 drop 起正则化作用，可提升模型鲁棒性
3. 推理时可动态调整 drop 率平衡速度与质量
4. 与 KV 缓存配合使用效果更佳

## 代表工作
- [[WEAVER]]: 使用 SPRINT（drop 概率 0.5）加速 928M 参数动力学 Transformer 训练

## 相关概念
- [[Transformer]]
- [[流匹配]]
- [[扩散强制]]
