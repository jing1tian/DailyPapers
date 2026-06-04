---
type: concept
aliases: [选择策略, 候选动作选择, Choice Policy Branch]
created: 2026-06-04
---

# Choice Policy

## 定义

一种 VLA 训练机制：在 VLM 输出端附加 N 个动作查询 token，通过线性头预测 N 个候选动作块，同时用评分头预测各候选的误差分数，以最小误差候选作为扩散模型的去噪起点。

## 数学形式

候选动作预测：

$$
\bar{a}_t^{(n)} = g_{act}(H_a)_{t,n}, \quad n = 1, \ldots, N
$$

Choice Policy 损失（取 N 个候选中最优的）：

$$
\mathcal{L}_{choice} = \frac{1}{B}\sum_{b=1}^{B} \min_n \frac{1}{T_b D}\|\hat{A}_b^{(n)} - A_b^*\|_1
$$

## 核心要点

1. **多假设避免陷阱**: 生成 N 个候选覆盖多种可能动作，最小化最优候选的误差而非均值
2. **评分头辅助选择**: 额外的 Score Query 头预测各候选的 MAE 误差，用于在推理时选择最优候选初始化扩散
3. **与扩散联合优化**: Choice Policy 提供初始化，扩散变换器精化输出，两者在表示空间协同训练
4. **关键贡献**: ERVLA 消融显示去除 Choice Policy 使 LIBERO-Plus 从 96.2% 降至 70.8%

## 代表工作

- [[ERVLA]]: 首次在具身 CoT VLA 框架中引入 Choice Policy + 知识截断 KV 条件化的组合

## 相关概念

- [[Action Chunking]]
- [[Diffusion Transformer (DiT)]]
- [[Flow Matching]]
- [[KV-Cache Conditioning]]
