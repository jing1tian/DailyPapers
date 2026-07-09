---
type: concept
aliases: [混合注意力, 跨专家注意力, Multi-Expert Attention]
---

# Mixed Attention

## 定义

混合注意力（Mixed Attention）是一种多专家并行架构中的注意力机制，将来自不同专家分支（如视频专家、动作专家、几何专家）的 token 拼接后共享注意力计算，各专家拥有独立的 Query/Key/Value 投影，并通过掩码矩阵控制跨分支的信息流动。

## 数学形式

$$X = [X_{e_1}, X_{e_2}, \ldots, X_{e_n}]$$

$$\text{Attn}(Q_e, K, V) = \text{softmax}\left(\frac{Q_e K^\top}{\sqrt{d}} + M_e\right) V$$

其中 $M_e$ 为专家 $e$ 的掩码矩阵，$-\infty$ 表示禁止读取的位置。

## 核心要点

1. **独立投影**: 每个专家使用独立的 $W_Q^e, W_K^e, W_V^e$，避免特征空间混淆
2. **掩码控制**: 通过掩码矩阵精细控制跨专家的信息读取权限，防止信息捷径（如动作 token 直接读取未来视频预测）
3. **推理时可移除**: 若某专家仅训练时使用，推理时直接从注意力序列中移除对应 token，不影响其余分支

## 代表工作

- [[MECo-WAM]]: 三路专家（视频/动作/4D 几何）共享混合注意力，衰减掩码控制几何分支的临时读取
- [[GeoSem-WAM]]: 多分支辅助预测头与主干共享表示
- [[Mixture-of-Transformers]]: 通用多专家 Transformer 架构

## 相关概念

- [[Mixture-of-Experts]]
- [[World Action Model]]
- [[Knowledge Distillation]]
