---
type: concept
aliases: [因果时序注意力, 因果时序多头注意力, Causal MHA, 单向时序注意力]
---

# Causal Temporal Attention

## 定义

在多帧历史特征序列上施加因果掩码（下三角掩码）的时序多头自注意力机制，每一帧只能关注自身及其之前的帧，不允许未来信息泄露。

## 数学形式

$$
\Delta\mathbf{x}_{v,p,t} = \text{Up}\!\left(\text{MHA}_{\text{causal}}\!\left(\mathbf{z}_{v,p,1:T}\right)_t\right)
$$

其中因果掩码确保位置 $t$ 的注意力权重满足 $\alpha_{t,j} = 0$ for $j > t$。

## 核心要点

1. **单向性**：通过下三角掩码矩阵实现，防止未来帧信息提前泄露
2. **帧间差分**：计算当前帧相对于历史帧的特征增量 $\Delta\mathbf{x}$，捕捉运动信息
3. **维度压缩**：通常在低维空间（Down 投影后）计算注意力，再 Up 投影回原维度
4. **插入位置**：插入视觉骨干中间层而非最终层效果更佳（保留更多局部特征）

## 代表工作

- [[ReflexVLA]]：在 ViT 中间层引入因果时序注意力，实现多帧运动感知，比交叉注意力或最终层插入提升更多

## 相关概念

- [[Self-Attention]]
- [[Transformer]]
- [[Block Causal Attention]]
- [[Time-Causal Attention]]
