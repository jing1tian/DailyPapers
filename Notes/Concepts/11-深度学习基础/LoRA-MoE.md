---
type: concept
aliases: [LoRA MoE, LoRA Mixture-of-Experts, 混合专家LoRA]
---

# LoRA-MoE

## 定义

将 [[LoRA]]（低秩适配）与 [[Mixture-of-Experts]]（混合专家）结合的参数高效微调方法：用多组 LoRA 矩阵作为"专家"，通过动态路由选择并加权激活，使模型对不同输入条件（任务、技能、领域）产生差异化的参数更新。

## 数学形式

$$
\begin{aligned}
\mathbf{p} &= \operatorname{softmax}(\mathbf{W}_{\mathrm{gate}}\mathbf{h} + \mathbf{b}^{(\ell)}) \\
\mathcal{T} &= \operatorname{TopK}(\mathbf{p}, k) \\
\Delta\mathbf{h} &= \sum_{i \in \mathcal{T}} \bar{p}_i \mathbf{B}_i \mathbf{A}_i \mathbf{h}
\end{aligned}
$$

其中 $\mathbf{A}_i \in \mathbb{R}^{r \times d_{\mathrm{in}}}$，$\mathbf{B}_i \in \mathbb{R}^{d_{\mathrm{out}} \times r}$ 是第 $i$ 个专家的 LoRA 矩阵，$r$ 为 rank。

## 核心要点

1. **参数效率**: 相比全量 MoE，只有 LoRA 参数参与路由，大幅减少可训练参数。
2. **动态条件化**: 路由可由输入内容、任务 ID、技能类型等条件调控（如加性偏置 $\mathbf{b}^{(\ell)}$）。
3. **负载均衡**: 通常加辅助损失（auxiliary loss）防止所有 token 坍缩到少数专家。
4. **灵活组合**: 可在 Transformer 的注意力层或 FFN 层分别部署独立的 LoRA-MoE 集合。

## 代表工作

- [[MAST-VLA]]: 技能感知 LoRA-MoE，用技能查询 token 和路由偏置 bank 对 7 类机器人原语技能做差异化专家激活

## 相关概念

- [[LoRA]]
- [[Mixture-of-Experts]]
- [[Residual Adapter]]
