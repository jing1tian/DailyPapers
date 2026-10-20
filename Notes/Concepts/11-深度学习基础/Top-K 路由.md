---
type: concept
aliases: [Top-K Routing, TopK Gating, top-k门控]
---

# Top-K 路由

## 定义

[[混合专家|混合专家（MoE）]]中的**稀疏路由**机制：对每个 token 的隐状态计算路由分数，仅激活得分最高的 K 个专家（通常 K=1 或 K=2），未激活的专家不参与计算，实现条件计算。

## 数学形式

$$
\mathbf{w} = \operatorname{TopKSoftmax}(W_r \cdot \operatorname{LN}(\mathbf{H}))
$$

$$
\text{output} = \sum_{e \in \mathcal{E}_{\mathrm{top-K}}} w_e \cdot E_e(\mathbf{H})
$$

辅助平衡损失（防止路由崩溃）：

$$
\mathcal{L}_{\mathrm{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

其中 $f_i$ 为专家负载比例，$P_i$ 为路由概率均值。

## 核心要点

1. **K=1（Switch Transformer 风格）**：最稀疏，计算效率最高，但方差大
2. **K=2（GShard 风格）**：稳定性更好，专家利用率更高
3. **平衡损失**：防止所有 token 路由到少数专家，造成 expert collapse
4. **与 Softmax 路由对比**：Top-K 路由是离散选择（不可微），通常用 straight-through 估计训练

## 代表工作

- [[LAW]]: MoTE 中使用 K=1 路由（top-1），结合共享专家缓解崩溃

## 相关概念

- [[MoTE]]
- [[Mixture of Experts]]
- [[混合专家]]
